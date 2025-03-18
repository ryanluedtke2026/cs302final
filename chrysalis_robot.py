from robot_config import robots
import sys
import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import json
import pandas as pd

logging.basicConfig(level=logging.DEBUG, filename='simulation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

real = ti.f32
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

use_toi = False

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
x_inc = vec()
rotation_inc = scalar()
omega_inc = scalar()

head_id = 3
goal = vec()

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -9.8
friction = 1.0
penalty = 1e4
damping = 10

gradient_clip = 30
spring_omega = 30
default_actuation = 0.05

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
# spring_length = -1 means it is a joint
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_phase = scalar()
spring_actuation = scalar()
spring_stiffness = scalar()

n_sin_waves = 10

n_hidden = 32
weights1 = scalar()
bias1 = scalar()
hidden = scalar()
weights2 = scalar()
bias2 = scalar()
actuation = scalar()

learning_rate = 0.01

def n_input_states():
    return n_sin_waves + 6 * n_objects + 2

initialized = False

def allocate_fields():
    global initialized
    if not initialized:
        ti.root.dense(ti.i,
                    max_steps).dense(ti.j,
                                    n_objects).place(x, v, rotation,
                                                        rotation_inc, omega, v_inc,
                                                        x_inc, omega_inc)
        ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass,
                                            inverse_inertia)
        ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                            spring_length, spring_offset_a,
                                            spring_offset_b, spring_stiffness,
                                            spring_phase, spring_actuation)
        ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
        ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
        ti.root.dense(ti.i, n_hidden).place(bias1)
        ti.root.dense(ti.i, n_springs).place(bias2)
        ti.root.dense(ti.ij, (max_steps, n_springs)).place(actuation)
        ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
        ti.root.place(loss, goal)
        ti.root.lazy_grad()

        initialized = True


dt = 0.001
learning_rate = 1.0

@ti.kernel
def apply_open_loop_contol(t: ti.i32):
    for i in range(n_springs):
        phase = spring_phase[i]
        frequency = spring_omega
        actuation[t, i] = ti.sin(frequency * t * dt + phase)

@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - x[t, head_id]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 6 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i,
                                  j * 6 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 2] * v[t,
                                                                  j][0] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 3] * v[t,
                                                                  j][1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves +
                                  4] * rotation[t, j] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 5] * omega[t,
                                                                      j] * 0.05

        actuation += weights1[i, n_objects * 6 + n_sin_waves] * goal[None][0]
        actuation += weights1[i,
                              n_objects * 6 + n_sin_waves + 1] * goal[None][1]
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        act = 0.0
        for j in ti.static(range(n_hidden)):
            act += weights2[i, j] * hidden[t, j]
        act += bias2[i]
        act = ti.tanh(act)
        actuation[t, i] = act


@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
        inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] *
                                    (halfsize[i][0] * halfsize[i][0] +
                                     halfsize[i][1] * halfsize[i][1]))
        # ti.print(inverse_mass[i])
        # ti.print(inverse_inertia[i])


@ti.func
def to_world(t, i, rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)

    rela_pos = rot_matrix @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location, toi_input):
    # ti.print(toi)
    delta_v = impulse * inverse_mass[i]
    delta_omega = (location - x[t, i]).cross(impulse) * inverse_inertia[i]

    toi = ti.min(ti.max(0.0, toi_input), dt)

    ti.atomic_add(x_inc[t + 1, i], toi * (-delta_v))
    ti.atomic_add(rotation_inc[t + 1, i], toi * (-delta_omega))

    ti.atomic_add(v_inc[t + 1, i], delta_v)
    ti.atomic_add(omega_inc[t + 1, i], delta_omega)


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(4)):
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])

            # Apply impulse so that there's no sinking
            normal = ti.Vector([0.0, 1.0])
            tao = ti.Vector([1.0, 0.0])

            rn = rela_pos.cross(normal)
            rt = rela_pos.cross(tao)
            impulse_contribution = inverse_mass[i] + (rn) ** 2 * \
                                   inverse_inertia[i]
            timpulse_contribution = inverse_mass[i] + (rt) ** 2 * \
                                    inverse_inertia[i]

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.0
            timpulse = 0.0
            new_corner_x = corner_x + dt * corner_v
            toi = 0.0
            if rela_v_ground < 0 and new_corner_x[1] < ground_height:
                impulse = -(1 +
                            elasticity) * rela_v_ground / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(friction * impulse,
                                      ti.max(-friction * impulse, timpulse))
                    if corner_x[1] > ground_height:
                        toi = -(corner_x[1] - ground_height) / ti.min(
                            corner_v[1], -1e-3)

            apply_impulse(t, i, impulse * normal + timpulse * tao,
                          new_corner_x, toi)

            penalty = 0.0
            if new_corner_x[1] < ground_height:
                # apply penalty
                penalty = -dt * penalty * (
                    new_corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t, i, penalty * normal, new_corner_x, 0)


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a, vel_a, rela_a = to_world(t, a, spring_offset_a[i])
        pos_b, vel_b, rela_b = to_world(t, b, spring_offset_b[i])
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        act = actuation[t, i]

        is_joint = spring_length[i] == -1

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act)
        if is_joint:
            target_length = 0.0
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        if is_joint:
            rela_vel = vel_a - vel_b
            rela_vel_norm = rela_vel.norm() + 1e-1
            impulse_dir = rela_vel / rela_vel_norm
            impulse_contribution = inverse_mass[a] + \
              impulse_dir.cross(rela_a) ** 2 * inverse_inertia[
                                     a] + inverse_mass[b] + impulse_dir.cross(rela_b) ** 2 * \
                                   inverse_inertia[
                                     b]
            # project relative velocity
            impulse += rela_vel_norm / impulse_contribution * impulse_dir

        apply_impulse(t, a, -impulse, pos_a, 0.0)
        apply_impulse(t, b, impulse, pos_b, 0.0)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1,
                                  i] + dt * omega[t, i] + rotation_inc[t, i]


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t, head_id] - goal[None]).norm()


gui = ti.GUI('Rigid Body Simulation', (512, 512), background_color=0xFFFFFF)


def get_world_loc(i, t, offset):
    rot = rotation[t, i]
    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    pos_ti = np.array([x[t, i][0], x[t, i][1]])  # Convert Taichi vector to NumPy array
    offset_array = np.array([offset[0], offset[1]])  # Ensure offset is a NumPy array
    pos = pos_ti + np.dot(rot_matrix, offset_array)  # Use NumPy dot for matrix multiplication
    return pos[0], pos[1]  # Return as tuple for direct use

def forward(output=None, visualize=True):
    try:
        logging.debug("Starting forward simulation with output=%s, visualize=%s", output, visualize)
        initialize_properties()

        interval = vis_interval
        total_steps = steps
        if output:
            logging.debug("Output directory will be created for visualization")
            os.makedirs(f'rigid_body/{output}/', exist_ok=True)
            interval = output_vis_interval
            total_steps *= 2

        goal[None] = [0.9, 0.15]
        for t in range(1, total_steps):
            nn1(t - 1)
            nn2(t - 1)
            # apply_open_loop_contol(t-1)
            collide(t - 1)
            apply_spring_force(t - 1)
            if use_toi:
                advance_toi(t)
            else:
                advance_no_toi(t)

            if (t + 1) % (interval * 2) == 0 and visualize:
                for i in range(n_objects):
                    points = []
                    for k in range(4):
                        offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                        rot = rotation[t, i]
                        rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                            [math.sin(rot),
                                                math.cos(rot)]])

                        halfsize_vector = np.array([halfsize[i][0], halfsize[i][1]])  # Convert Taichi vector to numpy array
                        pos = x[t, i] + rot_matrix @ (offset_scale * halfsize_vector)

                        points.append((pos[0], pos[1]))

                    for k in range(4):
                        gui.line(points[k],
                                points[(k + 1) % 4],
                                color=0x0,
                                radius=2)

                for i in range(n_springs):

                    pt1 = get_world_loc(spring_anchor_a[i], t, spring_offset_a[i])
                    pt2 = get_world_loc(spring_anchor_b[i], t, spring_offset_b[i])

                    color = 0xFF2233

                    if spring_actuation[i] != 0 and spring_length[i] != -1:
                        a = actuation[t - 1, i] * 0.5
                        color = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

                    if spring_length[i] == -1:
                        # logging.debug("Drawing line from %s to %s with color %s and radius %s", pt1, pt2, color)

                        gui.line(pt1, pt2, color=0x000000, radius=9)
                        gui.line(pt1, pt2, color=color, radius=7)
                    else:
                        gui.line(pt1, pt2, color=0x000000, radius=7)
                        gui.line(pt1, pt2, color=color, radius=5)

                gui.line((0.05, ground_height - 5e-3),
                        (0.95, ground_height - 5e-3),
                        color=0x0,
                        radius=5)

                file = None
                if output:
                    file = f'rigid_body/{output}/{t:04d}.png'
                    gui.show(file=file)
                else: 
                    gui.show()

        loss[None] = 0
        compute_loss(steps - 1)
        logging.debug("Completed forward simulation without error")
        clear_states()

    except Exception as e:
        logging.error("Simulation failed with exception: %s", e, exc_info=True)
        clear_states()
        raise RuntimeError("Simulation failed") from e
    

@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            x_inc[t, i] = ti.Vector([0.0, 0.0])
            rotation_inc[t, i] = 0.0
            omega_inc[t, i] = 0.0


"""
my lab 3 robot
"""

def add_object(objects, x, halfsize, rotation=0):
    objects.append([x, halfsize, rotation])
    return len(objects) - 1


# actuation 0.0 will be translated into default actuation
def add_spring(springs, a, b, offset_a, offset_b, length, stiffness, actuation=0.0):
    springs.append([a, b, offset_a, offset_b, length, stiffness, actuation])

# ADAPTED FROM GIVEN LEG CODE
def myRotAlong(length, angle_degrees, start_pos):
    """Rotate a point around another point by a given angle and distance."""
    angle_radians = math.radians(angle_degrees)
    new_x = start_pos[0] + length * math.cos(angle_radians)
    new_y = start_pos[1] + length * math.sin(angle_radians)
    return [new_x, new_y]

def build_robot_skeleton_vii(branching_factor=8, segment_length_range=(0.025, 0.075), angle_range=180, joint_stiffness=200, branch_spring_stiffness=50):
    objects = []
    springs = []
    body = add_object(objects, [0.5, 0.5], [0.1, 0.1])  # Main body of the robot

    min_length, max_length = segment_length_range
    segment_lengths = np.linspace(max_length, min_length, branching_factor)

    def create_segs(parent_id, base_angle, factor):
        parent_id_pos = np.array(objects[parent_id][0])
        parent_id_size = np.array(objects[parent_id][1])

        last_branch_id = None
        last_branch_center = None
        angle_increment = angle_range / factor
        starting_angle = 180

        for i in range(factor):
            new_angle = base_angle + starting_angle + i * angle_increment  # Evenly space segments around the circle
            segment_length = segment_lengths[i]
            joint_on_parent = myRotAlong(parent_id_size[0], new_angle, parent_id_pos)
            seg_center = myRotAlong(segment_length, new_angle, joint_on_parent)
            seg_id = add_object(objects, seg_center, [segment_length, segment_length / 2], math.radians(new_angle))

            add_spring(springs, parent_id, seg_id,
                       (myRotAlong(parent_id_size[0], new_angle, [0, 0])),
                       [-segment_length, 0],
                       -1, joint_stiffness)
            
            if last_branch_id is not None:
                spring_length = np.linalg.norm(np.array(seg_center) - np.array(last_branch_center))
                add_spring(springs, seg_id, last_branch_id, [0,0], [0,0], spring_length, branch_spring_stiffness)

            last_branch_id = seg_id
            last_branch_center = seg_center

    create_segs(body, 0, branching_factor)  # Start creating segments at angle 0

    return objects, springs, body


def setup_robot(objects, springs, h_id):
    global head_id
    head_id = h_id
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)
    allocate_fields()

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i][0]
        halfsize[i] = objects[i][1]
        rotation[0, i] = objects[i][2]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_offset_a[i] = s[2]
        spring_offset_b[i] = s[3]
        spring_length[i] = s[4]
        spring_stiffness[i] = s[5]
        if s[6]:
            spring_actuation[i] = s[6]
        else:
            spring_actuation[i] = default_actuation


def log_simulation_params(output_path, data):
    folder_path = 'evolution_json_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_output_path = os.path.join(folder_path, output_path)
    with open(folder_output_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def optimize(toi=True, visualize=True, output=None):
    global use_toi
    use_toi = toi

    min_loss = float('inf')

    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states())) * 0.5

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 1
    '''
  if visualize:
    clear_states()
    forward('initial{}'.format(robot_id))
  '''

    losses = []
    for iter in range(5):
        clear_states()

        # curr_visualize = visualize if iter == 19 else False

        with ti.ad.Tape(loss):
            if visualize:
                forward(output=output, visualize=True)
            else:
                forward(output=None, visualize=False)

        print('Iter=', iter, 'Loss=', loss[None])

        if np.isnan(loss[None]):
            print("NaN detected")
            return None, float('inf')

        losses.append(loss[None])
        if loss[None] < min_loss:
            min_loss = loss[None]

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(n_springs):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2

        print(total_norm_sqr)

        gradient_clip = 0.2
        scale = learning_rate * min(
            1.0, gradient_clip / (total_norm_sqr**0.5 + 1e-4))
        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_springs):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]

    # final_loss = losses[-1] if not np.isnan(losses[-1]) else np.nan
    #avg_loss = sum(losses) / len(losses)
    print(f'min loss found {min_loss}')
    return losses, min_loss

"""INITIALIZE MUTATION RATES FOR IMPROVED MUTATION METRICS PER EACH PARAMETER"""
mutation_rates = {
    'stiffness': 50,
    'branching': 2,
    'segment_length': 0.005
}
performance_history = {
    'stiffness': [],
    'branching': [],
    'segment_length': []
}

base_parameters = {
    'stiffness': 300,
    'branching': 7,
    'segment_length_range': (0.02, 0.09)
}

def cool_mutation_rate(initial_rate, generation, total_generations, cooling_factor=0.95):
    # Cooling factor < 1.0 reduces the mutation rate over generations
    # Adjust the cooling factor based on desired rate of cooling
    cooled_rate = initial_rate * (cooling_factor ** (generation / total_generations))
    return max(cooled_rate, 0.01)  # Ensure mutation rate doesn't go below a minimum threshold


def adapt_mutation_rate(history, generation, total_generations):
    # Apply cooling to each mutation rate
    for param in mutation_rates.keys():
        initial_rate = mutation_rates[param]
        if len(history[param]) >= 3:  # Ensure there is enough data to make a decision
            recent_trend = history[param][-3:]
            if all(recent < recent_trend[0] for recent in recent_trend[1:]):  # Check if recent changes are improvements
                # Decrease mutation rate more rapidly for improvements
                mutation_rates[param] = cool_mutation_rate(initial_rate, generation, total_generations, cooling_factor=0.85)
            else:
                # Otherwise, cool at a normal rate
                mutation_rates[param] = cool_mutation_rate(initial_rate, generation, total_generations)
        else:
            # Default cooling if not enough history
            mutation_rates[param] = cool_mutation_rate(initial_rate, generation, total_generations)

def mutate_parameters(performance_history, mutation_rates, base_params, generation, total_generations):
    adapt_mutation_rate(performance_history, generation, total_generations)  # Ensure rates are adapted based on history

    new_params = base_params.copy()
    new_params['stiffness'] = max(0, base_params['stiffness'] + np.random.uniform(-mutation_rates['stiffness'], mutation_rates['stiffness']))
    new_params['branching'] = max(3, base_params['branching'] + np.random.randint(-mutation_rates['branching'], mutation_rates['branching'] + 1))
    new_segment_length_range = (
        max(0.01, base_params['segment_length_range'][0] + np.random.uniform(-mutation_rates['segment_length'], mutation_rates['segment_length'])),
        max(0.01, base_params['segment_length_range'][1] + np.random.uniform(-mutation_rates['segment_length'], mutation_rates['segment_length']))
    )
    new_params['segment_length_range'] = new_segment_length_range

    return new_params


def evaluate_robot(params, generation, idx, max_attempts=3):
    attempt = 0
    min_loss = float('inf')

    while attempt < max_attempts:
        stiffness = params['stiffness']
        branching = params['branching']
        segment_length_range = params['segment_length_range']
        output_path = f'generation{generation}_robot{idx}_params.json'
        output_dir = f'generation{generation}_robot{idx}'

        objects, springs, head_id = build_robot_skeleton_vii(branching_factor=branching, joint_stiffness=stiffness, segment_length_range=segment_length_range)
        setup_robot(objects, springs, head_id)
        print(f'robot with stiffness: {stiffness} branching: {branching} segment_length_range: {segment_length_range}')
        try:
            if generation == 1:
                losses, temp_min_loss = optimize(toi=True, visualize=True, output=output_dir)
            else:
                losses, temp_min_loss = optimize(toi=True, visualize=False, output=output_dir)
            if losses is None:
                attempt += 1
                continue
            min_loss = temp_min_loss
            break
        except Exception as e:
            print(f"Error during simulation with params stiffness={stiffness}, branching={branching}: {str(e)}")
            attempt += 1
            if attempt >= max_attempts:
                print(f"All attempts failed for robot generation {generation} index {idx}, final attempt with inf loss.")
                break

        finally:
            log_simulation_params(output_path, {
                'generation': generation,
                'robot_index': idx,
                'params': {'stiffness': stiffness, 'branching': branching, 'segment_length_range': segment_length_range},
                'min_loss': min_loss
            })

            clear_states()

    return min_loss

def evolutionary_optimization(base_params, generations=8, population_size=5):
    current_params = base_params.copy()  # Correctly using dictionary copy
    best_params = base_params.copy()
    best_loss = float('inf')
    all_gen_losses = []

    for g in range(generations):
        print(f"Generation {g + 1}")

        adapt_mutation_rate(performance_history, g, generations)

        population = [mutate_parameters(performance_history, mutation_rates, current_params, g, generations) for _ in range(population_size)]
        gen_losses = []

        for idx, params in enumerate(population):
            final_loss = evaluate_robot(params, g + 1, idx)
            gen_losses.append(final_loss)

            # Track performance to adjust mutation rates
            performance_history['stiffness'].append(params['stiffness'])
            performance_history['branching'].append(params['branching'])
            performance_history['segment_length'].append(params['segment_length_range'])

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params.copy()  # Safe since it's a dictionary

        current_params = best_params.copy()  # Prepare for next generation
        all_gen_losses.append(gen_losses)
        print(f"End of Generation {g + 1}, Best Loss: {best_loss}")

    plot_gen_losses(all_gen_losses)
    return best_params, best_loss


def load_simulation_data(folder_path='evolution_json_data'):
    all_data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                data_point = json.loads(line)
                # Flatten the nested 'params' dictionary
                flat_data = {
                    'generation': data_point['generation'],
                    'robot_index': data_point['robot_index'],
                    'stiffness': data_point['params']['stiffness'],
                    'branching': data_point['params']['branching'],
                    'segment_length_range_min': data_point['params']['segment_length_range'][0],
                    'segment_length_range_max': data_point['params']['segment_length_range'][1],
                    'min_loss': data_point['min_loss']
                }
                all_data.append(flat_data)
    return pd.DataFrame(all_data)

def plot_parameter_impact(dataframe, save_path='parameter_impact.png'):
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))  # Adjust subplot for readability

    parameters = [
        ('stiffness', 'Impact of Stiffness on Loss', 'Stiffness'),
        ('branching', 'Impact of Branching on Loss', 'Branching'),
        ('segment_length_range_min', 'Impact of Segment Length Min on Loss', 'Segment Length Min'),
        ('segment_length_range_max', 'Impact of Segment Length Max on Loss', 'Segment Length Max')
    ]

    for i, (param, title, xlabel) in enumerate(parameters):
        # Ensure that data for plotting is clean and finite
        clean_data = dataframe.dropna(subset=[param, 'min_loss'])
        clean_data = clean_data[np.isfinite(clean_data[param]) & np.isfinite(clean_data['min_loss'])]

        axs[i].scatter(clean_data[param], clean_data['min_loss'], alpha=0.5)
        axs[i].set_title(title)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel('Loss')
        axs[i].grid(True)

        # Fit and plot a linear regression line if data is sufficient
        if len(clean_data) > 1:
            try:
                coeffs = np.polyfit(clean_data[param], clean_data['min_loss'], 1)
                poly = np.poly1d(coeffs)
                sorted_param = np.linspace(clean_data[param].min(), clean_data[param].max(), num=100)
                axs[i].plot(sorted_param, poly(sorted_param), "r--")  # Red dashed trend line
            except Exception as e:
                print(f"Failed to fit a model for {param}: {str(e)}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved as {save_path}.")


def plot_gen_losses(all_gen_losses):
    # Figure setup to handle two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Subplot 1: Losses for each individual per generation
    for i, losses in enumerate(all_gen_losses, 1):
        axs[0].plot(range(1, len(losses) + 1), losses, marker='o', label=f'Generation {i}')
    axs[0].set_title('Loss Evolution Per Individual Across Generations')
    axs[0].set_xlabel('Individual')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Minimum loss trend across generations
    min_losses = [min(losses) for losses in all_gen_losses]
    axs[1].plot(range(1, len(min_losses) + 1), min_losses, marker='o', linestyle='-', color='r')
    axs[1].set_title('Minimum Loss Evolution Across Generations')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Minimum Loss')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('evolutionary_loss_trends.png')
    plt.close()
    print("Comprehensive plot saved as 'evolutionary_loss_trends.png'.")

def main():
    #setup_new_robot()
    best_stiffness, best_loss = evolutionary_optimization(base_parameters)
    data = load_simulation_data()
    plot_parameter_impact(data)

if __name__ == '__main__':
    main()

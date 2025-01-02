import numpy as np
from scipy.optimize import least_squares

def normalize_points(theta, points, center_x, center_y):
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    transformed_points = np.dot(R, (points - [center_x, center_y]).T).T
    return transformed_points


def denormalize_points(theta, points, center_x, center_y):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    transformed_points = np.dot(R, points.T).T + [center_x, center_y]
    return transformed_points


def get_index(index, offset, n, out_indices):
    while index in out_indices:
        index = (index + offset) % n
    return index


def smooth_scale(vector):
    for i in range(vector.shape[1]):
        lengths = vector[:, i]
        zero_index = np.where(lengths == 0)[0]
        lengths_no_zeros = lengths[lengths != 0]
        if len(lengths_no_zeros) <= (len(lengths) - 2) / 2.0:
            lengths_no_zeros = np.zeros_like(lengths_no_zeros)
            final_lengths = np.zeros_like(lengths)
        else:

            Q1 = np.percentile(np.abs(lengths_no_zeros), 25)
            Q3 = np.percentile(np.abs(lengths_no_zeros), 75)
            IQR = Q3 - Q1

            if IQR < 0.2:
                IQR = IQR + 0.2

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            n = len(lengths_no_zeros)
            out_indices = \
                np.where(np.logical_or(np.abs(lengths_no_zeros) < lower_bound, np.abs(lengths_no_zeros) > upper_bound))[
                    0]
            final_lengths = np.zeros_like(lengths)

            if len(out_indices) == 0:
                median = np.median(np.abs(lengths_no_zeros))
                MAD = np.median(np.abs(np.abs(lengths_no_zeros) - median))
                if MAD < 0.1:
                    MAD = MAD + 0.1
                m_z_scores = 0.6745 * (np.abs(lengths_no_zeros) - median) / MAD
                out_indices = np.where(np.abs(m_z_scores) > 2.5)[0]

            for index in out_indices:
                left_index = get_index(index, -1, n, out_indices)
                right_index = get_index(index, 1, n, out_indices)

                left_length = np.abs(lengths_no_zeros[left_index])
                right_length = np.abs(lengths_no_zeros[right_index])

                avg_length = ((left_length + right_length) / 2)
                lengths_no_zeros[index] = avg_length * lengths_no_zeros[index] / np.abs(lengths_no_zeros[index])

        final_lengths[np.setdiff1d(np.arange(len(lengths)), zero_index)] = lengths_no_zeros

        vector[:, i] = final_lengths

    return vector


def offset_transfer(points, center_x, center_y, theta0, theta_1, delta_theta, offset, scale):
    R = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                  [np.sin(delta_theta), np.cos(delta_theta)]])
    transformed_points = np.dot(R, (points - [center_x, center_y]).T).T
    theta0 = theta0 + delta_theta
    transformed_points = transformed_points + [center_x, center_y]
    R_0 = np.array([[np.cos(theta0), np.sin(theta0)],
                    [-np.sin(theta0), np.cos(theta0)]])
    R_1 = np.array([[np.cos(theta_1), -np.sin(theta_1)],
                    [np.sin(theta_1), np.cos(theta_1)]])
    rotated_offset = np.dot(R_1, offset)

    rotated_offset = rotated_offset * [scale[2] if rotated_offset[0] < 0 else scale[3],
                                       scale[1] if rotated_offset[1] < 0 else scale[0]]
    rotated_offset_next = np.dot(R_0, rotated_offset)
    transformed_points = transformed_points + rotated_offset_next
    return transformed_points


def calculate_params(points):
    P1, P2 = points[0], points[-1]
    arc_points = points[1:-1]

    center_x, center_y = (P1 + P2) / 2
    major_axis_length = np.linalg.norm(P2 - P1)
    theta = np.arctan2(P2[1] - P1[1], P2[0] - P1[0])
    transformed_points = normalize_points(theta, arc_points, center_x, center_y)

    if np.max(np.abs(transformed_points[:, 1])) <= 2:
        minor_axis_length = 0

    else:
        def residuals(b):
            x, y = transformed_points[:, 0], transformed_points[:, 1]
            return (x / (major_axis_length / 2)) ** 2 + (y / b) ** 2 - 1

        minor_axis_length = 2 * least_squares(residuals, x0=1).x[0]

    return center_x, center_y, major_axis_length, minor_axis_length, theta


def post_process(driving_lm, driving_lm_next, ref_lm, ref_lm_next):
    for j in range(driving_lm.shape[1]):
        for i in range(driving_lm.shape[0]):
            if driving_lm[i, j] * ref_lm[i, j] < 0 and driving_lm_next[i, j] != 0:
                ref_lm_next[i, j] = -ref_lm_next[i, j] 
            elif driving_lm[i, j] == 0 and driving_lm_next[i, j] * ref_lm[i, j] < 0:
                ref_lm_next[i, j] = ref_lm_next[i, j] + ref_lm[i, j]
    return ref_lm_next


def processing_params(parts, lms):
    infos = []
    for part in parts:
        lm = lms[:, part, :]
        params = []
        for l in lm:
            xc, yc, a, b, theta = calculate_params(l)
            params.append([xc, yc, a, b, theta])
        infos.append({
            'params': params,
        })
    return infos


def get_offset_from_line(point, center, theta):
    translated_point = point - center

    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    transformed_point = np.dot(rotation_matrix, translated_point)

    return transformed_point


def process_boundary(lms, infos, part_index):
    offset = [get_offset_from_line(np.array([infos[i]['params'][0][0], infos[i]['params'][0][1]]),
                                   np.array([infos[0]['params'][0][0], infos[0]['params'][0][1]]),
                                   infos[0]['params'][0][4]) for i in range(len(part_index))]
    kp = [np.array([lms[0, 0, 0], lms[0, 0, 1]]), np.array([lms[0, 8, 0], lms[0, 8, 1]]),
          np.array([lms[0, 16, 0], lms[0, 16, 1]])]
    kp_offset = [get_offset_from_line(p, np.array([infos[0]['params'][0][0], infos[0]['params'][0][1]]),
                                      infos[0]['params'][0][4]) for p in kp]
    return offset + kp_offset


def process_boundary_scale(driving_offset, ref_offset):
    scale_pairs = [
        [((3, 1), (1, 1)), ((-2, 1), (1, 1)), ((-3, 0), (1, 0)), ((-1, 0), (1, 0))],
        [((3, 1), (2, 1)), ((-2, 1), (2, 1)), ((-3, 0), (2, 0)), ((-1, 0), (2, 0))],
        [((-4, 1), (3, 1)), ((1, 1), (3, 1)), ((-3, 0), (3, 0)), ((-1, 0), (3, 0))],
        [((3, 1), (4, 1)), ((-2, 1), (4, 1)), ((-3, 0), (4, 0)), ((-1, 0), (4, 0))],
        [((3, 1), (5, 1)), ((-2, 1), (5, 1)), ((-3, 0), (5, 0)), ((-1, 0), (5, 0))],
        [((-6, 1), (6, 1)), ((-4, 1), (6, 1)), ((-3, 0), (6, 0)), ((8, 0), (6, 0))],
        [((-6, 1), (7, 1)), ((-4, 1), (7, 1)), ((-3, 0), (7, 0)), ((8, 0), (7, 0))],
        [((-5, 1), (8, 1)), ((-4, 1), (8, 1)), ((6, 0), (8, 0)), ((-1, 0), (8, 0))],
        [((-5, 1), (9, 1)), ((-4, 1), (9, 1)), ((6, 0), (9, 0)), ((-1, 0), (9, 0))],
        [((-2, 1), (10, 1)), ((6, 1), (10, 1)), ((-3, 0), (10, 0)), ((11, 0), (10, 0))],
        [((-2, 1), (11, 1)), ((8, 1), (11, 1)), ((10, 0), (11, 0)), ((-1, 0), (11, 0))]
    ]

    def calc_ratio(i1, j1, i2, j2):
        return (np.abs(ref_offset[i1][j1] - ref_offset[i2][j2]) /
                np.abs(driving_offset[i1][j1] - driving_offset[i2][j2]))

    scales = []
    for pairs in scale_pairs:
        s = [calc_ratio(i1, j1, i2, j2) for ((i1, j1), (i2, j2)) in pairs]
        scales.append(s)

    scales.append([1, 1, 1, 1])

    return scales
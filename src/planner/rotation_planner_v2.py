import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

## From path_plan_4.py ##

# Helper function for cubic Bezier curve interpolation
def bezier_curve(t, p0, p1, p2, p3):
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# Generate curved path points
def generate_bezier_path(start_pos, end_pos, control1, control2, n_points):
    t_values = np.linspace(0, 1, n_points)
    curve_points = [bezier_curve(t, start_pos, control1, control2, end_pos) for t in t_values]
    return np.array(curve_points)


# Helper function for quaternion interpolation with max angle constraint
def lerp_quaternions(start_quat, end_quat, max_angle_deg):
    # Calculate the total angle between the quaternions
    rot_diff = R.from_quat(end_quat) * R.from_quat(start_quat).inv()
    total_angle_rad = rot_diff.magnitude()
    max_angle_rad = np.deg2rad(max_angle_deg)

    # Calculate the minimum steps required
    min_steps = int(np.ceil(total_angle_rad / max_angle_rad))

    quaternions = []
    for i in range(min_steps + 1):
        t = i / min_steps
        interp_quat = (1 - t) * start_quat + t * end_quat
        interp_quat /= np.linalg.norm(interp_quat)  # Normalize to keep it valid
        quaternions.append(interp_quat)
    return np.array(quaternions), min_steps


# Helper function to calculate initial orientation based on first movement direction
def calculate_initial_orientation(first_pos, second_pos):
    # Compute initial direction vector
    direction = second_pos - first_pos
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # Define the desired orientation: align z-axis (forward) with the direction vector
    z_axis = direction  # New forward direction
    x_axis = np.cross([0, -1, 0], z_axis)  # Cross with y-down to get the right direction
    x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis
    y_axis = np.cross(z_axis, x_axis)  # y-axis is orthogonal to both

    # Create rotation matrix from these axes
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Convert rotation matrix to Euler angles
    initial_rotation = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    return initial_rotation


# Function to generate a smooth trajectory of poses
def smoothen_trajectory(start_pose, end_pose, positions, max_angle_deg=10, gravity_vector=[0, 0, -1]):
    """
    Generates a smooth trajectory with position and orientation transitions between start and end poses.

    Parameters:
    -----------
    start_pose : np.array
        An array representing the starting pose, formatted as 4x4 matrix
        with position in meters and rotation angles in degrees.

    end_pose : np.array
        An array representing the ending pose, formatted as 4x4 matrix
        with position in meters and rotation angles in degrees.

    positions : np.array
        A 2D array of shape (n_points, 3) containing 3D coordinates along the trajectory in meters.
        Each row represents a position [x, y, z] in the global coordinate system.

    max_angle_deg : float, optional, default=10
        Maximum allowable rotation angle in degrees between each interpolated step.
        This determines the smoothness of orientation transitions along the path.

    gravity_vector : list or np.array, optional, default=[0, 0, -1]
        A vector representing the gravity direction, used to align the y-axis of the orientation.
        Defaults to downward in the z-axis. Should be a normalized vector if non-standard gravity is applied.

    Returns:
    --------
    np.array
        A 2D array of shape (n_points, 6) where each row represents a pose [x, y, z, roll, pitch, yaw].
        Positions are based on the input `positions`, and rotations are smoothly interpolated based on
        `start_pose` and `end_pose` with the specified constraints.

    Notes:
    ------
    - This function is designed to handle various trajectory lengths. It dynamically assigns orientation
      stages based on input parameters and constraints such as `max_angle_deg`.
    - If the trajectory length is greater than twice the `steps_for_180_deg` value, the function uses
      three stages: start, middle, and end orientation adjustments. Otherwise, it applies simpler
      orientation logic based on the number of positions.

    Raises:
    -------
    ValueError
        If the number of positions and rotations does not match any specified case logic,
        an error is raised to indicate a potential issue with the input values or logic flow.
    """

    # Separate position and orientation
    _, start_rot = start_pose[:3], start_pose[3:]
    _, end_rot = end_pose[:3], end_pose[3:]

    steps_for_180_deg = int(180 / max_angle_deg)  # Half-circle interpolation

    # Convert start and end rotations to quaternions
    start_quat = R.from_euler('xyz', start_rot, degrees=True).as_quat()
    end_quat = R.from_euler('xyz', end_rot, degrees=True).as_quat()

    # Interpolate quaternions with max angle constraint
    interp_quats, min_rotation_steps = lerp_quaternions(start_quat, end_quat, max_angle_deg)
    num_of_points = len(positions)
    num_of_rotations = min_rotation_steps + 1

    if num_of_points > steps_for_180_deg * 2:
        assert (num_of_points > num_of_rotations)

    print("Number of positions:", num_of_points)
    print("Number of rotations (min steps):", num_of_rotations)

    # Initialize rotations array
    rotations = np.zeros((max(num_of_points, num_of_rotations), 3))

    # Case 1: More rotations than positions
    if num_of_rotations > num_of_points:
        # interp_quats, _ = lerp_quaternions(start_quat, end_quat, max_angle_deg)
        rotations = R.from_quat(interp_quats).as_euler('xyz', degrees=True)

        extra_points = num_of_rotations - num_of_points
        print("More rotations than positions; extending positions with end point by", extra_points)
        positions = np.vstack([positions, np.tile(positions[-1], (extra_points, 1))])

    # Case 2: Positions > steps_for_180_deg * 2 (Positions should exceed rotations)
    elif num_of_points > steps_for_180_deg * 2:
        print("Positions exceed 36; setting trajectory-following rotations in the middle.")

        # Calculate split counts based on max_angle_deg to ensure smooth interpolation
        start_count = steps_for_180_deg
        end_count = steps_for_180_deg
        print("Start count:", start_count)
        print("End count:", end_count)

        # Stage 1: Interpolate from start_quat to the first trajectory-following rotation
        # Calculate the direction vector using positions[start_count] and positions[start_count + 1]
        direction = positions[start_count + 1] - positions[start_count]
        direction /= np.linalg.norm(direction)

        # Create rotation matrix for the start direction and align y-axis with gravity
        z_axis = direction
        y_axis = gravity_vector
        x_axis = np.cross(y_axis, z_axis)  # Ensure orthogonal x-axis
        x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis
        y_axis = np.cross(z_axis, x_axis)  # Recompute y to ensure orthogonality

        target_rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        target_quat = R.from_matrix(target_rotation_matrix).as_quat()

        # Interpolate from start_quat to target_quat to smoothly transition into trajectory-following rotation
        start_interp_quats, _ = lerp_quaternions(start_quat, target_quat, max_angle_deg)
        # print("Start interp quats shape:", start_interp_quats.shape)
        rotations[:len(start_interp_quats)] = R.from_quat(start_interp_quats).as_euler('xyz', degrees=True)

        # Stage 2: Interpolate from the last trajectory-following rotation to end_rot
        # Calculate the direction vector using positions[-end_count - 1] and positions[-end_count]
        end_direction = positions[-end_count] - positions[-end_count - 1]
        end_direction /= np.linalg.norm(end_direction)

        # Create rotation matrix for end direction and align y-axis with gravity
        z_axis = end_direction
        y_axis = gravity_vector
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        end_target_rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        end_target_quat = R.from_matrix(end_target_rotation_matrix).as_quat()

        # Interpolate from the last trajectory-following rotation to the end_rot smoothly
        end_interp_quats, _ = lerp_quaternions(end_target_quat, end_quat, max_angle_deg)
        # print("End interp quats shape:", end_interp_quats.shape)
        rotations[-len(end_interp_quats):] = R.from_quat(end_interp_quats).as_euler('xyz', degrees=True)

        # Calculate the middle section range
        middle_start = len(start_interp_quats)
        middle_end = num_of_points - len(end_interp_quats)
        # print("Middle start index:", middle_start)
        # print("Middle end index:", middle_end)

        # Stage 3: Apply trajectory-following rotations for the middle section
        for i in range(middle_start, middle_end):
            direction = positions[i + 1] - positions[i]
            direction /= np.linalg.norm(direction)

            # Create rotation matrix to align z-axis with direction and y-axis with gravity
            z_axis = direction
            y_axis = gravity_vector
            x_axis = np.cross(y_axis, z_axis)  # Ensure orthogonal x-axis
            x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis
            y_axis = np.cross(z_axis, x_axis)  # Recompute y to ensure orthogonality

            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            rotations[i] = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    # Case 3: Positions exceed rotations
    elif num_of_rotations <= num_of_points:
        print("Positions exceed rotations; splitting rotations at start and end with identity in center.")

        # Calculate split point, adjusting for odd num_of_rotations
        start_count = num_of_rotations // 2
        end_count = num_of_rotations - start_count

        # Apply rotations at the start
        rotations[:start_count] = R.from_quat(interp_quats[:start_count]).as_euler('xyz', degrees=True)

        # Set center rotations to match the last rotation at start boundary
        center_rotation = R.from_quat(interp_quats[start_count]).as_euler('xyz', degrees=True)
        rotations[start_count: num_of_points - end_count] = center_rotation

        # Apply rotations at the end
        rotations[-end_count:] = R.from_quat(interp_quats[-end_count:]).as_euler('xyz', degrees=True)

    else:
        raise NotImplementedError("This case logic has not been implemented yet. Please review inputs or logic.")

    # Combine positions and rotations into poses
    poses = np.hstack((positions, rotations))
    # print("Positions shape:", positions.shape)
    # print("Rotations shape:", rotations.shape)
    return poses


def poses_to_transformation_matrices(poses: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Converts an Nx6 array of positions and Euler angles into an Nx4x4 array of transformation matrices.

    Args:
        poses (np.ndarray): An (N, 6) array where each row contains [x, y, z, roll, pitch, yaw].
        degrees (bool, optional): If True, rotation angles are assumed to be in degrees.
                                  If False, angles are in radians. Default is True.

    Returns:
        np.ndarray: An (N, 4, 4) array of 4x4 transformation matrices.
    """
    n_poses = poses.shape[0]
    
    # Initialize the Nx4x4 transformation matrix array
    transformation_matrices = np.zeros((n_poses, 4, 4))
    
    for i in range(n_poses):
        # Extract position and rotation (Euler angles)
        position = poses[i, :3]  # [x, y, z]
        euler_angles = poses[i, 3:]  # [roll, pitch, yaw]
        
        # Create rotation matrix from Euler angles
        rotation_matrix = R.from_euler('xyz', euler_angles, degrees=degrees).as_matrix()
        
        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position
        
        # Assign to the output array
        transformation_matrices[i] = transformation_matrix
    
    return transformation_matrices

def smoothen_trajectory_v2(start_pose, end_pose, positions, max_angle_deg=10, gravity_vector=[0, 0, -1]):
    """
    Generates a smooth trajectory with position and orientation transitions between start and end poses.

    Parameters:
    -----------
    start_pose : np.array
        An array representing the starting pose, formatted as 4x4 matrix
        with position in meters and rotation angles in degrees.

    end_pose : np.array
        An array representing the ending pose, formatted as 4x4 matrix
        with position in meters and rotation angles in degrees.

    positions : np.array
        A 2D array of shape (n_points, 3) containing 3D coordinates along the trajectory in meters.
        Each row represents a position [x, y, z] in the global coordinate system.

    max_angle_deg : float, optional, default=10
        Maximum allowable rotation angle in degrees between each interpolated step.
        This determines the smoothness of orientation transitions along the path.

    gravity_vector : list or np.array, optional, default=[0, 0, -1]
        A vector representing the gravity direction, used to align the y-axis of the orientation.
        Defaults to downward in the z-axis. Should be a normalized vector if non-standard gravity is applied.

    Returns:
    --------
    np.array
        A 2D array of shape (n_points, 4, 4) where each row represents a pose.
        Positions are based on the input `positions`, and rotations are smoothly interpolated based on
        `start_pose` and `end_pose` with the specified constraints.

    Notes:
    ------
    - This function is designed to handle various trajectory lengths. It dynamically assigns orientation
      stages based on input parameters and constraints such as `max_angle_deg`.
    - If the trajectory length is greater than twice the `steps_for_180_deg` value, the function uses
      three stages: start, middle, and end orientation adjustments. Otherwise, it applies simpler
      orientation logic based on the number of positions.

    Raises:
    -------
    ValueError
        If the number of positions and rotations does not match any specified case logic,
        an error is raised to indicate a potential issue with the input values or logic flow.
    """
    # Extract positions and orientations from start and end poses
    # start_position = start_pose[:3, 3]
    start_rotation = R.from_matrix(start_pose[:3, :3])
    # end_position = end_pose[:3, 3]
    end_rotation = R.from_matrix(end_pose[:3, :3])

    # Convert start and end rotations to quaternions for interpolation
    start_quat = start_rotation.as_quat()
    end_quat = end_rotation.as_quat()

    steps_for_180_deg = int(180 / max_angle_deg)  # Half-circle interpolation

    # # Separate position and orientation
    # _, start_rot = start_pose[:3], start_pose[3:]
    # _, end_rot = end_pose[:3], end_pose[3:]

    # steps_for_180_deg = int(180 / max_angle_deg)  # Half-circle interpolation

    # # Convert start and end rotations to quaternions
    # start_quat = R.from_euler('xyz', start_rot, degrees=True).as_quat()
    # end_quat = R.from_euler('xyz', end_rot, degrees=True).as_quat()

    # Interpolate quaternions with max angle constraint
    interp_quats, min_rotation_steps = lerp_quaternions(start_quat, end_quat, max_angle_deg)
    num_of_points = len(positions)
    num_of_rotations = min_rotation_steps + 1

    if num_of_points > steps_for_180_deg * 2:
        assert (num_of_points > num_of_rotations)

    print("Number of positions:", num_of_points)
    print("Number of rotations (min steps):", num_of_rotations)

    # Initialize rotations array
    rotations = np.zeros((max(num_of_points, num_of_rotations), 3))

    # Case 1: More rotations than positions
    if num_of_rotations > num_of_points:
        # interp_quats, _ = lerp_quaternions(start_quat, end_quat, max_angle_deg)
        rotations = R.from_quat(interp_quats).as_euler('xyz', degrees=True)

        extra_points = num_of_rotations - num_of_points
        print("More rotations than positions; extending positions with end point by", extra_points)
        positions = np.vstack([positions, np.tile(positions[-1], (extra_points, 1))])

    # Case 2: Positions > steps_for_180_deg * 2 (Positions should exceed rotations)
    elif num_of_points > steps_for_180_deg * 2:
        print("Positions exceed 36; setting trajectory-following rotations in the middle.")

        # Calculate split counts based on max_angle_deg to ensure smooth interpolation
        start_count = steps_for_180_deg
        end_count = steps_for_180_deg
        print("Start count:", start_count)
        print("End count:", end_count)

        # Stage 1: Interpolate from start_quat to the first trajectory-following rotation
        # Calculate the direction vector using positions[start_count] and positions[start_count + 1]
        direction = positions[start_count + 1] - positions[start_count]
        direction /= np.linalg.norm(direction)

        # Create rotation matrix for the start direction and align y-axis with gravity
        z_axis = direction
        y_axis = gravity_vector
        x_axis = np.cross(y_axis, z_axis)  # Ensure orthogonal x-axis
        x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis
        y_axis = np.cross(z_axis, x_axis)  # Recompute y to ensure orthogonality

        target_rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        target_quat = R.from_matrix(target_rotation_matrix).as_quat()

        # Interpolate from start_quat to target_quat to smoothly transition into trajectory-following rotation
        start_interp_quats, _ = lerp_quaternions(start_quat, target_quat, max_angle_deg)
        # print("Start interp quats shape:", start_interp_quats.shape)
        rotations[:len(start_interp_quats)] = R.from_quat(start_interp_quats).as_euler('xyz', degrees=True)

        # Stage 2: Interpolate from the last trajectory-following rotation to end_rot
        # Calculate the direction vector using positions[-end_count - 1] and positions[-end_count]
        end_direction = positions[-end_count] - positions[-end_count - 1]
        end_direction /= np.linalg.norm(end_direction)

        # Create rotation matrix for end direction and align y-axis with gravity
        z_axis = end_direction
        y_axis = gravity_vector
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        end_target_rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        end_target_quat = R.from_matrix(end_target_rotation_matrix).as_quat()

        # Interpolate from the last trajectory-following rotation to the end_rot smoothly
        end_interp_quats, _ = lerp_quaternions(end_target_quat, end_quat, max_angle_deg)
        # print("End interp quats shape:", end_interp_quats.shape)
        rotations[-len(end_interp_quats):] = R.from_quat(end_interp_quats).as_euler('xyz', degrees=True)

        # Calculate the middle section range
        middle_start = len(start_interp_quats)
        middle_end = num_of_points - len(end_interp_quats)
        # print("Middle start index:", middle_start)
        # print("Middle end index:", middle_end)

        # Stage 3: Apply trajectory-following rotations for the middle section
        for i in range(middle_start, middle_end):
            direction = positions[i + 1] - positions[i]
            direction /= np.linalg.norm(direction)

            # Create rotation matrix to align z-axis with direction and y-axis with gravity
            z_axis = direction
            y_axis = gravity_vector
            x_axis = np.cross(y_axis, z_axis)  # Ensure orthogonal x-axis
            x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis
            y_axis = np.cross(z_axis, x_axis)  # Recompute y to ensure orthogonality

            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            rotations[i] = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    # Case 3: Positions exceed rotations
    elif num_of_rotations <= num_of_points:
        print("Positions exceed rotations; splitting rotations at start and end with identity in center.")

        # Calculate split point, adjusting for odd num_of_rotations
        start_count = num_of_rotations // 2
        end_count = num_of_rotations - start_count

        # Apply rotations at the start
        rotations[:start_count] = R.from_quat(interp_quats[:start_count]).as_euler('xyz', degrees=True)

        # Set center rotations to match the last rotation at start boundary
        center_rotation = R.from_quat(interp_quats[start_count]).as_euler('xyz', degrees=True)
        rotations[start_count: num_of_points - end_count] = center_rotation

        # Apply rotations at the end
        rotations[-end_count:] = R.from_quat(interp_quats[-end_count:]).as_euler('xyz', degrees=True)

    else:
        raise NotImplementedError("This case logic has not been implemented yet. Please review inputs or logic.")

    # Combine positions and rotations into poses
    poses = np.hstack((positions, rotations))
    # print("Positions shape:", positions.shape)
    # print("Rotations shape:", rotations.shape)
    poses = poses_to_transformation_matrices(poses)
    # import torch
    # if torch.stack([torch.isnan(torch.from_numpy(i)).any() for i in poses]).any():
    #     print("debug")
    return poses


# Visualization function with orientation
def visualize_trajectory_with_orientation(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot start and end poses
    ax.scatter(*poses[0][:3], color='green', label='Start Pose')
    ax.scatter(*poses[-1][:3], color='red', label='End Pose')

    # Plot the interpolated trajectory
    x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]
    ax.plot(x, y, z, color='blue', label='Interpolated Path')

    # Add orientation quivers to represent rotation
    # for i in range(0, len(poses), max(1, len(poses) // 10)):  # Adjust to show fewer quivers
    for i in range(len(poses)):  # Plot quivers at each position
        pos = poses[i][:3]
        rot = poses[i][3:]
        # Compute direction vectors based on rotation
        rot_matrix = R.from_euler('xyz', rot, degrees=True).as_matrix()

        # Define quiver arrows for each axis
        for j, color in zip(range(3), ['r', 'g', 'b']):  # RGB for XYZ axis directions
            ax.quiver(pos[0], pos[1], pos[2],
                      rot_matrix[0, j], rot_matrix[1, j], rot_matrix[2, j],
                      length=0.5, color=color, normalize=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Test parameters
    start_pose = np.array([0, 0, 0, -90, 0, 0])  # Start position and rotation (x, y, z, roll, pitch, yaw)
    end_pose = np.array([1, 1, 1, -90, 0, -45])  # End position and rotation

    # Generate positions along a Bezier curve
    n_points = 40  # Number of points to generate, adjust as needed
    control1 = np.array([3, 15, 5])  # First control point for Bezier curve
    control2 = np.array([8, 5, 10])  # Second control point for Bezier curve
    start_pos, start_rot = start_pose[:3], start_pose[3:]
    end_pos, end_roend_rott = end_pose[:3], end_pose[3:]
    positions = generate_bezier_path(start_pose[:3], end_pose[:3], control1, control2, n_points + 1)

    # Generate and visualize poses
    poses = smoothen_trajectory(start_pose, end_pose, positions, max_angle_deg=10, gravity_vector=[0, 0, -1])
    visualize_trajectory_with_orientation(poses)
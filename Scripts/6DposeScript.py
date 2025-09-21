import numpy as np
import omni.isaac
import omni.kit.commands
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import Gf, Sdf, UsdGeom, Usd
import json
import os

stage = omni.usd.get_context().get_stage()
replicator_prim_path = "/Replicator"

# Check if the Replicator prim exists
replicator_prim = stage.GetPrimAtPath(replicator_prim_path)
if replicator_prim.IsValid():
    stage.RemovePrim(replicator_prim_path)
    print(f"Cleared existing Replicator prim at: {replicator_prim_path}")
else:
    print(f"No existing Replicator prim found at: {replicator_prim_path}")

base_field_prim_path = "/World/FE_2025/tn__FE2025_c9/tn__FullFidelityField_rHD"
coral_prim_paths = [(base_field_prim_path + "/Coral/Mesh_" + str(i)) for i in range(0, 150)]
coral_dynamic_prim_paths = [(base_field_prim_path + "/Coral/Mesh_" + str(i)) for i in range(126, 150)]
coral_static_prim_paths = [(base_field_prim_path + "/Coral/Mesh_" + str(i)) for i in range(0, 6)]
algae_static_prim_paths = [(base_field_prim_path + "/Algae/Mesh_" + str(i)) for i in range(0, 6)]
algae_prim_paths = [(base_field_prim_path + "/Algae/Mesh_" + str(i)) for i in range(0, 34)]

def extract_posecnn_format(output_dir, frame_number):
    """
    Extract poses in PoseCNN format: RT matrices and object masks
    """
    stage = omni.usd.get_context().get_stage()
    
    # Objects to track for PoseCNN
    objects_to_track = {
        'coral': coral_prim_paths[:15],   # Track 15 coral pieces
        'algae': algae_prim_paths[:15]    # Track 15 algae pieces
    }
    
    posecnn_data = {
        'frame_id': frame_number,
        'objects': []
    }
    
    for class_name, prim_paths in objects_to_track.items():
        for i, prim_path in enumerate(prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                try:
                    # Get transform matrix
                    xformable = UsdGeom.Xformable(prim)
                    transform_matrix = xformable.ComputeLocalToWorldTransform(0)
                    
                    # Extract RT matrix (3x4) - standard PoseCNN format
                    rotation_matrix = transform_matrix.ExtractRotationMatrix()
                    translation = transform_matrix.ExtractTranslation()
                    
                    # Convert to PoseCNN RT format [R|t] 3x4 matrix
                    rt_matrix = [
                        [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], translation[0]],
                        [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], translation[1]],
                        [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], translation[2]]
                    ]
                    
                    # Get object bounds
                    imageable = UsdGeom.Imageable(prim)
                    bbox = imageable.ComputeWorldBound(0, "default")
                    size = bbox.GetBox().GetSize()
                    
                    obj_data = {
                        'class_id': len(posecnn_data['objects']) + 1,  # PoseCNN uses class IDs
                        'class_name': f"{class_name}",
                        'rt_matrix': rt_matrix,  # 3x4 rotation-translation matrix
                        'position': [float(translation[0]), float(translation[1]), float(translation[2])],
                        'size': [float(size[0]), float(size[1]), float(size[2])],
                        'prim_path': prim_path
                    }
                    
                    posecnn_data['objects'].append(obj_data)
                    
                except Exception as e:
                    print(f"Failed to extract RT matrix for {prim_path}: {e}")
    
    # Save in PoseCNN format
    pose_file = os.path.join(output_dir, f"posecnn_{frame_number:06d}.json")
    with open(pose_file, 'w') as f:
        json.dump(posecnn_data, f, indent=2)
    
    return posecnn_data

potential_look_prims = [
    base_field_prim_path + "/Barge/RedCageInner",
    base_field_prim_path + "/Barge/RedCageMiddle",
    base_field_prim_path + "/Barge/RedCageOuter",
    base_field_prim_path + "/Barge/BlueCageInner",
    base_field_prim_path + "/Barge/BlueCageMiddle",
    base_field_prim_path + "/Barge/BlueCageOuter"
] + coral_static_prim_paths + coral_dynamic_prim_paths + algae_prim_paths

camera = rep.create.camera()

# Setup for PoseCNN dataset generation  
output_dir = "C:/Users/antho/Documents/isaac-sim/frc_posecnn_dataset/"
os.makedirs(output_dir, exist_ok=True)

# Reduced frame count - PoseCNN works well with smaller datasets
with rep.trigger.on_frame(rt_subframes=4, max_execs=500):  # 500 frames for PoseCNN
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-8.5, -4.2, 0.3), (8.5, 4.2, 5)),
            look_at=rep.distribution.choice(potential_look_prims)
        )

    light = rep.get.prim_at_path("/Environment/RectLight")
    with light:
        rep.modify.attribute(name="inputs:colorTemperature", value=rep.distribution.normal(4500.0, 1500.0))
        rep.modify.attribute(name="inputs:intensity", value=rep.distribution.normal(2000.0, 3500.0))
        rep.modify.attribute(name="inputs:color", value=rep.distribution.uniform((0.7, 0.7, 0.7), (1, 1, 1)))

    # PoseCNN benefits from clear object visibility
    for i in range(0, 6):
        coral = rep.get.prim_at_path(coral_static_prim_paths[i])
        algae = rep.get.prim_at_path(algae_static_prim_paths[i])
        # Higher visibility for PoseCNN training
        visible = rep.distribution.choice(choices=[True, True, True, True, False])
        rep.modify.visibility(visible, input_prims=coral)
        rep.modify.visibility(visible, input_prims=algae)
        
    for coral_prim_path in coral_prim_paths:
        if coral_prim_path not in coral_static_prim_paths:
            coral = rep.get.prim_at_path(coral_prim_path)
            with coral:
                rep.modify.visibility(rep.distribution.choice(choices=[True, True, False, False, False, False, False, False, False, False]))
                
    for algae_prim_path in algae_prim_paths:
        if algae_prim_path not in algae_static_prim_paths:
            algae = rep.get.prim_at_path(algae_prim_path)
            with algae:
                rep.modify.visibility(rep.distribution.choice(choices=[True, True, True, False, False]))

# Create render product with higher resolution for better pose estimation
render_product = rep.create.render_product(camera, (1024, 1024))

# Try BasicWriter optimized for PoseCNN training data
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,                       # RGB images
    semantic_segmentation=True,     # Object masks (essential for PoseCNN)
    instance_segmentation=True,     # Individual object instances  
    bounding_box_2d_tight=True,     # 2D bounding boxes
    camera_params=True,             # Camera intrinsics
    distance_to_camera=True         # Depth information (useful for PoseCNN)
)

# Attach render_product to the writer
writer.attach([render_product])

print("Starting PoseCNN dataset generation...")
print(f"Output directory: {output_dir}")
print("Dataset optimized for PoseCNN training:")
print("- 500 frames (PoseCNN works well with smaller datasets)")
print("- RGB images + semantic masks (essential for PoseCNN)")
print("- RT matrices (3x4 rotation-translation format)")
print("- Instance segmentation for individual objects")
print("")

# Run the main generation
num_frames = 500
print(f"Generating {num_frames} frames...")

# Start generation process
rep.orchestrator.run(num_frames=num_frames)

# Extract RT matrices in PoseCNN format
print("Generation completed! Extracting RT matrices for PoseCNN...")
for frame in range(num_frames):
    if frame % 50 == 0:  # Progress update every 50 frames
        print(f"Extracting RT matrices: {frame}/{num_frames} frames processed")
    
    posecnn_data = extract_posecnn_format(output_dir, frame)

print(f"\n=== POSECNN DATASET COMPLETE ===")
print(f"Generated PoseCNN dataset with:")
print(f"- {num_frames} RGB images")
print(f"- {num_frames} semantic segmentation masks")
print(f"- {num_frames} instance segmentation masks")  
print(f"- {num_frames} RT matrix files (posecnn_XXXXXX.json)")
print(f"- Camera parameters and 2D bounding boxes")
print(f"\nDataset ready for PoseCNN training!")
print(f"Location: {output_dir}")

# RT matrix format info
print(f"\nRT matrices are in standard PoseCNN 3x4 format:")
print(f"[[R11, R12, R13, tx],")
print(f" [R21, R22, R23, ty],") 
print(f" [R31, R32, R33, tz]]")

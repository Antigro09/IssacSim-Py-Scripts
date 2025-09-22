import numpy as np
import omni.usd
import omni.replicator.core as rep

stage = omni.usd.get_context().get_stage()
replicator_prim_path = "/Replicator"

# Clear existing Replicator prim if it exists
replicator_prim = stage.GetPrimAtPath(replicator_prim_path)
if replicator_prim.IsValid():
    stage.RemovePrim(replicator_prim_path)
    print(f"Cleared existing Replicator prim at: {replicator_prim_path}")
else:
    print(f"No existing Replicator prim found at: {replicator_prim_path}")

# Define your object prim paths (example coral and algae)
base_field_prim_path = "/World/FE_2025/tn__FE2025_c9/tn__FullFidelityField_rHD"
coral_static_prim_paths = [(base_field_prim_path + "/Coral/Mesh_" + str(i)) for i in range(0, 6)]
algae_static_prim_paths = [(base_field_prim_path + "/Algae/Mesh_" + str(i)) for i in range(0, 6)]

# Prims to look at for camera targeting
potential_look_prims = [
    base_field_prim_path + "/Barge/RedCageInner",
    base_field_prim_path + "/Barge/RedCageMiddle",
    base_field_prim_path + "/Barge/RedCageOuter",
    base_field_prim_path + "/Barge/BlueCageInner",
    base_field_prim_path + "/Barge/BlueCageMiddle",
    base_field_prim_path + "/Barge/BlueCageOuter"
] + coral_static_prim_paths + algae_static_prim_paths

camera = rep.create.camera()
with rep.trigger.on_frame(rt_subframes=4):
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-8.5, -4.2, 0.3), (8.5, 4.2, 5)),
            look_at=rep.distribution.choice(potential_look_prims)
        )


    # Randomize lighting
    light = rep.get.prim_at_path("/Environment/RectLight")
    if light:
        with light:
            rep.modify.attribute(name="inputs:colorTemperature", value=rep.distribution.normal(4500.0, 1500.0))
            rep.modify.attribute(name="inputs:intensity", value=rep.distribution.normal(0.0, 3500.0))
            rep.modify.attribute(name="inputs:color", value=rep.distribution.uniform((0.7, 0.7, 0.7), (1, 1, 1)))

    # Randomize visibility of static objects
    for coral_path in coral_static_prim_paths:
        coral = rep.get.prim_at_path(coral_path)
        rep.modify.visibility(rep.distribution.choice(choices=[True, False, False, False]), input_prims=coral)
    for algae_path in algae_static_prim_paths:
        algae = rep.get.prim_at_path(algae_path)
        rep.modify.visibility(rep.distribution.choice(choices=[True, False, False, False]), input_prims=algae)

# Render 512x512 RGB images from the camera
render_product = rep.create.render_product(camera, (512, 512))

# Initialize PoseWriter (outputs 6D pose + keypoints + bbox + segmentation etc)
pose_writer = rep.WriterRegistry.get("PoseWriter")
pose_writer.initialize(
    output_dir="C:\\User\\antho\\Documents\\Issac-Sim-Data\\",
    # Optionally specify object filtering, camera intrinsics, etc.
)
pose_writer.attach([render_product])

# Run the simulation graph for 5120 frames (adjust as needed)
rep.orchestrator.run(num_frames=5120)

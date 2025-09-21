import numpy as np
import omni.isaac
import omni.kit.commands
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import Gf, Sdf, UsdGeom, Usd

stage = omni.usd.get_context().get_stage()
replicator_prim_path = "/Replicator"

# Check if the Replicator prim exists
replicator_prim = stage.GetPrimAtPath(replicator_prim_path)
if replicator_prim.IsValid():
    # Remove the Replicator prim and its children
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

potential_look_prims = [
    base_field_prim_path + "/Barge/RedCageInner",
    base_field_prim_path + "/Barge/RedCageMiddle",
    base_field_prim_path + "/Barge/RedCageOuter",
    base_field_prim_path + "/Barge/BlueCageInner",
    base_field_prim_path + "/Barge/BlueCageMiddle",
    base_field_prim_path + "/Barge/BlueCageOuter"
] + coral_static_prim_paths + coral_dynamic_prim_paths + algae_prim_paths

camera = rep.create.camera()
with rep.trigger.on_frame(rt_subframes=4):
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-8.5, -4.2, 0.3), (8.5, 4.2, 5)),
            look_at=rep.distribution.choice(potential_look_prims)
        )

    light = rep.get.prim_at_path("/Environment/RectLight")
    with light:
        # scale_rand = rep.distribution.uniform(0.5, 1.5)
        # rep.modify.attribute(name="xformOp:scale", value=(scale_rand, scale_rand, scale_rand))
        rep.modify.attribute(name="inputs:colorTemperature", value=rep.distribution.normal(4500.0, 1500.0))
        rep.modify.attribute(name="inputs:intensity", value=rep.distribution.normal(0.0, 3500.0))
        rep.modify.attribute(name="inputs:color", value=rep.distribution.uniform((0.7, 0.7, 0.7), (1, 1, 1)))

    for i in range(0, 6):
        coral = rep.get.prim_at_path(coral_static_prim_paths[i])
        algae = rep.get.prim_at_path(algae_static_prim_paths[i])
        visible = rep.distribution.choice(choices=[True, False, False, False, False, False])
        rep.modify.visibility(visible, input_prims=coral)
        rep.modify.visibility(visible, input_prims=algae)
    for coral_prim_path in coral_prim_paths:
        if coral_prim_path not in coral_static_prim_paths:
            coral = rep.get.prim_at_path(coral_prim_path)
            with coral:
                rep.modify.visibility(rep.distribution.choice(choices=[True, False, False, False, False, False, False, False, False, False]))
    for algae_prim_path in algae_prim_paths:
        if algae_prim_path not in algae_static_prim_paths:
            algae = rep.get.prim_at_path(algae_prim_path)
            with algae:
                rep.modify.visibility(rep.distribution.choice(choices=[True, False, False, False, False]))
    

# Will render 512x512 images
render_product = rep.create.render_product(camera, (512, 512))

# Get a Kitti Writer and initialize its defaults
writer = rep.WriterRegistry.get("KittiWriter")
writer.initialize(
    output_dir=f"C:\\User\\Antho\\Documents\\issaxc-sim\\2dReefscape\\",
    bbox_height_threshold=5,
    fully_visible_threshold=0.75,
    omit_semantic_type=True
)
# Attach render_product to the writer
writer.attach([render_product])

# Run the simulation graph
rep.orchestrator.run(num_frames=5120)

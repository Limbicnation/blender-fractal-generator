bl_info = {
    "name": "Advanced Fractal Geometry Generator",
    "author": "Your Name",
    "version": (1, 2),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Fractal",
    "description": "Generate fractal-based geometry modifications including 3D fractals",
    "warning": "Processing large meshes may be slow",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import bmesh
import random
import math
import time
import traceback
from mathutils import Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    FloatVectorProperty,
    EnumProperty,
    BoolProperty
)

# Global constants for safety
MAX_ALLOWED_FACES = 10000  # Prevents processing too many faces at once
DEFAULT_BATCH_SIZE = 500  # Process faces in batches to avoid memory spikes
DEBUG = False  # Set to True for additional console output

def debug_print(message):
    """Print debug messages to console if DEBUG is enabled"""
    if DEBUG:
        print(f"[Fractal Generator] {message}")

def validate_face(face):
    """Check if a face is valid for processing"""
    if not face or not hasattr(face, "is_valid") or not face.is_valid:
        return False
    try:
        # Skip very degenerate faces
        if face.calc_area() < 0.00001:
            return False
        # Skip faces with too many vertices (can cause issues)
        if len(face.verts) > 100:
            return False
        # Check for valid normal
        if face.normal.length < 0.0001:
            return False
        return True
    except:
        return False

def ensure_bmesh_lookup_tables(bm):
    """Ensure all BMesh lookup tables are valid"""
    try:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        return True
    except:
        return False

def get_selected_faces_431(obj, max_faces=MAX_ALLOWED_FACES):
    """Special function for Blender 4.3.x to get selected faces with safety limits"""
    selected_faces = []
    debug_print(f"Using improved 4.3.x selection detection")
    bm = None
    
    try:
        # Force selection sync from mesh to BMesh
        if obj.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='EDIT')
            
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        ensure_bmesh_lookup_tables(bm)
        
        # More explicit selection check
        selected_faces = [f for f in bm.faces if f.select and validate_face(f)]
        debug_print(f"Direct selection check found {len(selected_faces)} faces")
        
        # If still empty, try alternative approach
        if not selected_faces:
            debug_print("Trying mesh polygon selection approach")
            # Get selection from mesh itself
            selected_polygon_indices = [p.index for p in mesh.polygons if p.select]
            selected_faces = [bm.faces[i] for i in selected_polygon_indices 
                              if i < len(bm.faces) and validate_face(bm.faces[i])]
            debug_print(f"Polygon approach found {len(selected_faces)} faces")
            
        # Final approach: If available, use selection history
        if not selected_faces and hasattr(bm, "select_history") and bm.select_history:
            debug_print("Trying selection history")
            for elem in bm.select_history:
                if isinstance(elem, bmesh.types.BMFace) and validate_face(elem):
                    selected_faces.append(elem)
            debug_print(f"History approach found {len(selected_faces)} faces")
        
        # Apply maximum face limit for safety
        if max_faces > 0 and len(selected_faces) > max_faces:
            debug_print(f"Limiting selection from {len(selected_faces)} to {max_faces} faces")
            selected_faces = selected_faces[:max_faces]
        
        # Log final result
        debug_print(f"Final selection count: {len(selected_faces)}")
        return selected_faces, bm
        
    except Exception as e:
        debug_print(f"Selection detection error: {e}")
        if DEBUG:
            traceback.print_exc()
        return [], bm

def safe_mode_set(mode, obj=None):
    """Safely change object mode with error handling"""
    try:
        if not obj:
            obj = bpy.context.active_object
        if obj and obj.mode != mode:
            bpy.ops.object.mode_set(mode=mode)
        return True
    except Exception as e:
        debug_print(f"Mode change error: {e}")
        return False

class FRACTAL_PT_main_panel(bpy.types.Panel):
    bl_label = "Fractal Generator"
    bl_idname = "FRACTAL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Fractal'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        wm = context.window_manager
        
        # Show progress during processing
        if hasattr(wm, 'fractal_is_processing') and wm.fractal_is_processing:
            box = layout.box()
            box.label(text="Processing Fractal...", icon='TIME')
            
            # Show progress if available
            if hasattr(wm, 'fractal_progress'):
                box.prop(wm, "fractal_progress", text="Progress")
            
            # Cancel button
            box.operator("mesh.fractal_cancel", text="Cancel", icon='X')
            return
        
        # Warning about performance
        if scene.fractal_iterations > 100 or scene.fractal_scale > 5.0:
            box = layout.box()
            box.alert = True
            box.label(text="Warning: High Performance Settings", icon='ERROR')
            box.label(text="Processing may be slow", icon='INFO')

        # Main Fractal Button
        row = layout.row(align=True)
        row.scale_y = 2.0  # Make button bigger
        row.operator("mesh.fractal_generate", text="Generate Fractal", icon='OUTLINER_OB_SURFACE')
        
        # Randomize Seed Button
        row = layout.row(align=True)
        row.operator("mesh.fractal_randomize_seed", text="Randomize Seed", icon='FILE_REFRESH')

        # Basic Settings
        box = layout.box()
        box.label(text="Basic Settings", icon='SETTINGS')
        box.prop(scene, "fractal_type")
        box.prop(scene, "fractal_scale")
        box.prop(scene, "fractal_complexity")
        box.prop(scene, "fractal_selected_only")
        box.prop(scene, "fractal_face_limit")
        
        # Stepping Pattern Settings
        box = layout.box()
        box.label(text="Stepping Pattern Settings", icon='MOD_ARRAY')
        
        # First Extrusion Settings
        sub_box = box.box()
        sub_box.label(text="First Extrusion", icon='DRIVER_TRANSFORM')
        sub_box.prop(scene, "fractal_first_extrude_amount")
        sub_box.prop(scene, "fractal_extrude_along_normal")
        
        # Inset Settings
        sub_box = box.box()
        sub_box.label(text="Inset Controls", icon='MOD_SOLIDIFY')
        sub_box.prop(scene, "fractal_inset_amount")
        sub_box.prop(scene, "fractal_inset_depth")
        sub_box.prop(scene, "fractal_inset_relative")
        sub_box.prop(scene, "fractal_inset_edges_only")
        
        # Second Extrusion Settings
        sub_box = box.box()
        sub_box.label(text="Second Extrusion", icon='EMPTY_ARROWS')
        sub_box.prop(scene, "fractal_second_extrude_factor")
        sub_box.prop(scene, "fractal_use_individual_normals")

        # Advanced Options
        box = layout.box()
        box.label(text="Advanced Settings", icon='MODIFIER')
        box.prop(scene, "fractal_iterations")
        box.prop(scene, "fractal_seed")
        box.prop(scene, "use_smooth_shading")
        
        # Safety Settings
        box = layout.box()
        box.label(text="Safety Settings", icon='CHECKMARK')
        box.prop(scene, "fractal_batch_processing")
        if scene.fractal_batch_processing:
            box.prop(scene, "fractal_batch_size")

class MESH_OT_fractal_randomize_seed(bpy.types.Operator):
    bl_idname = "mesh.fractal_randomize_seed"
    bl_label = "Randomize Seed"
    bl_description = "Generate a new random seed value"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Generate a new random seed
        new_seed = int((time.time() * 1000) % 10000) + 1
        context.scene.fractal_seed = new_seed
        self.report({'INFO'}, f"New random seed: {new_seed}")
        return {'FINISHED'}

class MESH_OT_fractal_cancel(bpy.types.Operator):
    bl_idname = "mesh.fractal_cancel"
    bl_label = "Cancel Fractal Generation"
    bl_description = "Cancel the current fractal generation process"
    
    def execute(self, context):
        context.window_manager.fractal_should_cancel = True
        self.report({'INFO'}, "Cancelling fractal generation...")
        return {'FINISHED'}

class MESH_OT_fractal_generate(bpy.types.Operator):
    bl_idname = "mesh.fractal_generate"
    bl_label = "Generate Fractal"
    bl_options = {'REGISTER', 'UNDO'}
    
    _timer = None
    current_batch = 0
    face_batches = []
    bm = None
    mesh = None
    obj = None
    original_mode = 'OBJECT'
    processed_faces = 0
    extruded_faces = 0
    start_time = 0
    completed = False

    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.type == 'MESH' and 
                not getattr(context.window_manager, 'fractal_is_processing', False))

    def process_complex_pattern(self, face, fractal_value, scene):
        """Process a face with extrude → insert → extrude pattern - the core of the fractal stepping effect"""
        try:
            # Skip if face is not valid
            if not validate_face(face):
                return False
                
            # Skip faces with very low fractal values
            if fractal_value < 0.1:
                return False
                
            face_normal = face.normal.copy()
            if face_normal.length < 0.001:
                # Use global Z if normal is invalid
                face_normal = Vector((0, 0, 1))
            else:
                face_normal.normalize()
                
            # --- STEP 1: FIRST EXTRUSION ---
            # Extrude the face
            result = bmesh.ops.extrude_face_region(self.bm, geom=[face])
            
            # Get newly created geometry
            new_geom = result["geom"]
            new_faces = [g for g in new_geom if isinstance(g, bmesh.types.BMFace)]
            new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]
            
            # Calculate first extrusion strength using base amount and fractal value
            extrude_strength = scene.fractal_first_extrude_amount * fractal_value * scene.fractal_complexity
            extrude_strength = min(extrude_strength, 2.0)  # Cap maximum extrusion
            
            # Generate a deterministic random factor based on face center and seed
            if scene.fractal_complexity > 0.5:
                # Use face's position and seed to create deterministic randomness
                face_center = face.calc_center_median()
                position_hash = (face_center.x * 1000 + face_center.y * 100 + face_center.z * 10) * scene.fractal_seed
                # Generate a pseudo-random factor that's consistent for the same face and seed
                rand_factor = 0.8 + ((position_hash % 400) / 1000)  # Range: 0.8 to 1.2
                extrude_strength *= rand_factor
            
            # Move vertices along normal for first extrusion
            if new_verts:
                if scene.fractal_extrude_along_normal:
                    # Use face normal for extrusion
                    extrude_vec = face_normal * extrude_strength
                else:
                    # Use global Z for extrusion
                    extrude_vec = Vector((0, 0, extrude_strength))
                    
                bmesh.ops.translate(
                    self.bm,
                    vec=extrude_vec,
                    verts=new_verts
                )
            
            # --- STEP 2: INSET FACES ---
            # Set up inset parameters
            inset_amount = scene.fractal_inset_amount * fractal_value
            inset_amount = min(inset_amount, 0.9)  # Cap maximum inset
            
            inset_depth = scene.fractal_inset_depth * fractal_value
            
            # Inset the newly created faces to create the stepping effect
            inset_result = bmesh.ops.inset_region(
                self.bm,
                faces=new_faces,
                thickness=inset_amount,
                depth=inset_depth,
                use_even_offset=True,
                use_interpolate=True,
                use_relative_offset=scene.fractal_inset_relative,
                use_edge_rail=scene.fractal_inset_edges_only
            )
            
            # Get the inner faces from the inset operation
            inner_faces = inset_result.get("faces", [])
            
            # --- STEP 3: SECOND EXTRUSION ---
            # Only proceed if we have valid inner faces
            if inner_faces:
                # For each inner face, extrude separately for better control
                for inner_face in inner_faces:
                    if not validate_face(inner_face):
                        continue
                    
                    # Determine which normal to use
                    if scene.fractal_use_individual_normals:
                        # Use this face's own normal (more variation)
                        inner_normal = inner_face.normal.copy()
                        if inner_normal.length < 0.001:
                            inner_normal = face_normal  # Fall back to original normal
                        else:
                            inner_normal.normalize()
                    else:
                        # Use original face normal (more uniform)
                        inner_normal = face_normal
                        
                    # Extrude this inner face (the second extrusion in our pattern)
                    face_extrude_result = bmesh.ops.extrude_face_region(self.bm, geom=[inner_face])
                    face_verts = [g for g in face_extrude_result["geom"] if isinstance(g, bmesh.types.BMVert)]
                    
                    if face_verts:
                        # The second extrusion is a factor of the first, creating the stepping effect
                        second_extrude_strength = extrude_strength * scene.fractal_second_extrude_factor
                        
                        if scene.fractal_extrude_along_normal:
                            # Use the determined normal for extrusion
                            second_extrude_vec = inner_normal * second_extrude_strength
                        else:
                            # Use global Z for extrusion
                            second_extrude_vec = Vector((0, 0, second_extrude_strength))
                            
                        bmesh.ops.translate(
                            self.bm,
                            vec=second_extrude_vec,
                            verts=face_verts
                        )
            
            return True
            
        except Exception as e:
            debug_print(f"Error in fractal stepping pattern: {e}")
            if DEBUG:
                traceback.print_exc()
            return False

    def get_fractal_value(self, center, fractal_type, scene):
        """Calculate fractal value based on type with safety checks"""
        try:
            scale = min(scene.fractal_scale, 10.0)  # Cap scale for safety
            iterations = min(scene.fractal_iterations, 200)  # Cap iterations for safety
            
            x = center.x * scale
            y = center.y * scale
            z = center.z * scale
            
            if fractal_type == 'MANDELBROT':
                return self.mandelbrot_value(x, y, iterations)
            elif fractal_type == 'JULIA':
                return self.julia_value(x, y, iterations)
            elif fractal_type == 'QUINTIC_MANDELBULB':
                return self.quintic_mandelbulb_value(x, y, z, iterations)
            elif fractal_type == 'CUBIC_MANDELBULB':
                return self.cubic_mandelbulb_value(x, y, z, iterations)
            else:
                return 0.5  # Default value for unknown types
        except:
            # Fallback value if calculation fails
            return 0.3
    
    def mandelbrot_value(self, x, y, max_iter):
        """Calculate Mandelbrot set value with early bailout optimization"""
        c = complex(x, y)
        z = 0
        
        # Early bailout check
        if x*x + y*y > 4.0:
            return 0.0  # Definitely outside the set
            
        # Main iteration loop with faster bailout
        for i in range(max_iter):
            z_abs = abs(z)
            if z_abs > 4:  # Use 4 instead of 2 for bailout (faster check)
                return i / max_iter
            z = z * z + c
            
            # Safety check for infinity/NaN
            if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                return 0.0
        
        return 1.0
    
    def julia_value(self, x, y, max_iter, seed=-0.8 + 0.156j):
        """Calculate Julia set value with safety checks"""
        z = complex(x, y)
        c = seed  # Fixed seed for consistent pattern
        
        # Early bailout
        if abs(z) > 4.0:
            return 0.0
            
        for i in range(max_iter):
            if abs(z) > 4:
                return i / max_iter
            z = z * z + c
            
            # Safety check for infinity/NaN
            if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                return 0.0
        
        return 1.0
    
    def quintic_mandelbulb_value(self, x, y, z, max_iter):
        """Calculate 3D Quintic Mandelbulb value with safety checks"""
        # Initialize point
        cx, cy, cz = x, y, z  # Original point (c)
        px, py, pz = 0, 0, 0  # Start at origin (p)
        
        # Main iteration loop
        power = 5  # Quintic power
        bailout = 4.0
        
        # Limit iterations for safety
        iterations = min(max_iter, 100)
        
        for i in range(iterations):
            # Calculate squared radius
            r2 = px*px + py*py + pz*pz
            
            # Early bailout check
            if r2 > bailout:
                return i / iterations
            
            # Check for NaN or inf
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
            
            # Avoid division by zero
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                r = math.sqrt(r2)
                theta = math.atan2(math.sqrt(px*px + py*py), pz)
                phi = math.atan2(py, px)
            
            # Calculate r^power (with safeguards against extreme values)
            r_pow = min(pow(r, power), 1000.0)
            
            # Calculate new point in spherical coords
            theta_new = theta * power
            phi_new = phi * power
            
            # Convert back to cartesian coords
            px = r_pow * math.sin(theta_new) * math.cos(phi_new) + cx
            py = r_pow * math.sin(theta_new) * math.sin(phi_new) + cy
            pz = r_pow * math.cos(theta_new) + cz
            
            # Check for NaN or inf in coordinates
            if (math.isnan(px) or math.isnan(py) or math.isnan(pz) or
                math.isinf(px) or math.isinf(py) or math.isinf(pz)):
                return 0.0
        
        return 0.0  # Inside the set
    
    def cubic_mandelbulb_value(self, x, y, z, max_iter):
        """Calculate 3D Cubic Mandelbulb value with safety checks"""
        # Similar to quintic but with power=3
        cx, cy, cz = x, y, z  # Original point (c)
        px, py, pz = 0, 0, 0  # Start at origin (p)
        
        power = 3  # Cubic power
        bailout = 4.0
        iterations = min(max_iter, 100)  # Limit iterations for safety
        
        for i in range(iterations):
            r2 = px*px + py*py + pz*pz
            
            if r2 > bailout:
                return i / iterations
            
            # Check for NaN or inf
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
            
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                r = math.sqrt(r2)
                theta = math.atan2(math.sqrt(px*px + py*py), pz)
                phi = math.atan2(py, px)
            
            # Safety cap for r_pow
            r_pow = min(pow(r, power), 1000.0)
            
            theta_new = theta * power
            phi_new = phi * power
            
            px = r_pow * math.sin(theta_new) * math.cos(phi_new) + cx
            py = r_pow * math.sin(theta_new) * math.sin(phi_new) + cy
            pz = r_pow * math.cos(theta_new) + cz
            
            # Check for NaN or inf in coordinates
            if (math.isnan(px) or math.isnan(py) or math.isnan(pz) or
                math.isinf(px) or math.isinf(py) or math.isinf(pz)):
                return 0.0
        
        return 0.0  # Inside the set

    def modal(self, context, event):
        """Modal handler for batch processing"""
        try:
            if event.type == 'ESC' or context.window_manager.fractal_should_cancel:
                self.report({'INFO'}, "Fractal generation cancelled")
                self.cleanup(context, cancelled=True)
                return {'CANCELLED'}
                
            if event.type == 'TIMER':
                # Update progress
                if hasattr(context.window_manager, 'fractal_progress'):
                    progress = (self.current_batch / len(self.face_batches)) * 100
                    context.window_manager.fractal_progress = progress
                
                # Process next batch
                if self.current_batch < len(self.face_batches):
                    self.process_batch(context, self.current_batch)
                    self.current_batch += 1
                    return {'RUNNING_MODAL'}
                else:
                    # All batches processed
                    self.finish(context)
                    return {'FINISHED'}
                    
            return {'PASS_THROUGH'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error in modal: {str(e)}")
            self.cleanup(context, cancelled=True)
            return {'CANCELLED'}
    
    def process_batch(self, context, batch_idx):
        """Process a batch of faces with fractal operations"""
        try:
            # Get the batch
            batch = self.face_batches[batch_idx]
            scene = context.scene
            
            # Make sure we're in edit mode
            if self.obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
                self.bm = bmesh.from_edit_mesh(self.mesh)
                ensure_bmesh_lookup_tables(self.bm)
            
            # Process each face in batch
            for face in batch:
                # Skip if face is no longer valid
                if not validate_face(face):
                    continue
                
                # Calculate face center
                center = face.calc_center_median()
                
                # Calculate fractal value
                fractal_value = self.get_fractal_value(
                    center, scene.fractal_type, scene
                )
                
                # Apply extrusion based on fractal value
                if fractal_value > 0.1:  # Threshold to avoid tiny extrusions
                    try:
                        # Always use extrude → insert → extrude pattern
                        success = self.process_complex_pattern(face, fractal_value, scene)
                        if success:
                            self.extruded_faces += 1
                    except Exception as e:
                        debug_print(f"Error extruding face: {e}")
                
                self.processed_faces += 1
            
            # Update the mesh
            bmesh.update_edit_mesh(self.mesh)
            
            # Force redraw
            if context.area:
                context.area.tag_redraw()
                
            return True
            
        except Exception as e:
            debug_print(f"Error processing batch: {e}")
            if DEBUG:
                traceback.print_exc()
            return False
    
    def execute(self, context):
        """Main execute function"""
        try:
            # Initialize
            self.start_time = time.time()
            self.obj = context.active_object
            self.mesh = self.obj.data
            scene = context.scene
            
            # Initialize window manager properties
            wm = context.window_manager
            wm.fractal_is_processing = True
            wm.fractal_should_cancel = False
            wm.fractal_progress = 0.0
            
            # Set random seed
            if scene.fractal_seed > 0:
                random.seed(scene.fractal_seed)
            else:
                random.seed(None)
            
            # Store original mode and switch to edit mode
            self.original_mode = self.obj.mode
            if self.obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            
            # Get faces to process
            face_limit = min(scene.fractal_face_limit, MAX_ALLOWED_FACES)
            
            if scene.fractal_selected_only:
                selected_faces, bm = get_selected_faces_431(self.obj, face_limit)
                self.bm = bm
                
                if not selected_faces:
                    self.report({'WARNING'}, "No valid selected faces found. Select faces first.")
                    self.cleanup(context, cancelled=True)
                    return {'CANCELLED'}
                
                faces_to_process = selected_faces
            else:
                # Process all faces
                self.bm = bmesh.from_edit_mesh(self.mesh)
                ensure_bmesh_lookup_tables(self.bm)
                all_faces = [f for f in self.bm.faces if validate_face(f)]
                
                # Apply face limit
                faces_to_process = all_faces[:face_limit]
                
                if not faces_to_process:
                    self.report({'WARNING'}, "No valid faces found in mesh.")
                    self.cleanup(context, cancelled=True)
                    return {'CANCELLED'}
            
            # Report total faces
            self.report({'INFO'}, f"Processing {len(faces_to_process)} faces")
            
            # Reset counters
            self.processed_faces = 0
            self.extruded_faces = 0
            
            # Check for batch processing
            if scene.fractal_batch_processing:
                # Prepare batches
                batch_size = min(scene.fractal_batch_size, 1000)  # Safety cap
                
                # Create batches
                self.face_batches = []
                for i in range(0, len(faces_to_process), batch_size):
                    batch = faces_to_process[i:i+batch_size]
                    self.face_batches.append(batch)
                
                self.current_batch = 0
                
                # Start modal timer
                wm = context.window_manager
                self._timer = wm.event_timer_add(0.1, window=context.window)
                wm.modal_handler_add(self)
                
                return {'RUNNING_MODAL'}
                
            else:
                # Process all at once 
                self.face_batches = [faces_to_process]
                self.process_batch(context, 0)
                self.finish(context)
                return {'FINISHED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            self.cleanup(context, cancelled=True)
            return {'CANCELLED'}
    
    def finish(self, context):
        """Complete processing and clean up"""
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Apply smooth shading if enabled
        try:
            if context.scene.use_smooth_shading:
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.shade_smooth()
        except:
            pass
            
        # Clean up
        self.cleanup(context, cancelled=False)
        
        # Report success
        self.report({'INFO'}, f"Fractal generation completed: {self.extruded_faces}/{self.processed_faces} faces in {elapsed_time:.2f}s")
    
    def cleanup(self, context, cancelled=False):
        """Clean up resources"""
        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        
        # Reset window manager properties
        context.window_manager.fractal_is_processing = False
        context.window_manager.fractal_should_cancel = False
        context.window_manager.fractal_progress = 0.0
        
        # Try to return to original mode
        try:
            if cancelled and self.original_mode != 'EDIT':
                bpy.ops.object.mode_set(mode=self.original_mode)
        except:
            pass
        
        # Force redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

# Property registration functions
def register_properties():
    # Processing state properties
    bpy.types.WindowManager.fractal_is_processing = BoolProperty(
        default=False
    )
    bpy.types.WindowManager.fractal_should_cancel = BoolProperty(
        default=False
    )
    bpy.types.WindowManager.fractal_progress = FloatProperty(
        name="Progress",
        default=0.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE'
    )
    
    # Basic properties
    bpy.types.Scene.fractal_iterations = IntProperty(
        name="Iterations",
        description="Number of fractal iterations (higher = more detailed but slower)",
        default=50,
        min=5,
        max=200
    )
    bpy.types.Scene.fractal_scale = FloatProperty(
        name="Scale",
        description="Scale of the fractal pattern",
        default=1.5,
        min=0.1,
        max=10.0
    )
    bpy.types.Scene.fractal_complexity = FloatProperty(
        name="Complexity",
        description="Complexity of the fractal pattern (affects extrusion height)",
        default=0.5,
        min=0.1,
        max=2.0
    )
    bpy.types.Scene.fractal_seed = IntProperty(
        name="Random Seed",
        description="Seed for random generation (0 for system time)",
        default=1
    )
    bpy.types.Scene.fractal_type = EnumProperty(
        name="Fractal Type",
        description="Type of fractal pattern to generate",
        items=[
            ('MANDELBROT', "Mandelbrot", "Classic 2D Mandelbrot set"),
            ('JULIA', "Julia", "Classic 2D Julia set"),
            ('QUINTIC_MANDELBULB', "Quintic Mandelbulb", "3D Quintic Mandelbulb (power=5)"),
            ('CUBIC_MANDELBULB', "Cubic Mandelbulb", "3D Cubic Mandelbulb (power=3)"),
        ],
        default='MANDELBROT'
    )
    
    # Pattern properties for extrude-inset-extrude stepping effect
    bpy.types.Scene.fractal_extrude_along_normal = BoolProperty(
        name="Extrude Along Normals",
        description="Extrude geometry along face normals (rather than global Z)",
        default=True
    )
    
    bpy.types.Scene.fractal_use_individual_normals = BoolProperty(
        name="Use Individual Face Normals",
        description="Use each face's individual normal for extrusion (creates more variation)",
        default=True
    )
    
    # Complex pattern properties
    bpy.types.Scene.fractal_inset_amount = FloatProperty(
        name="Inset Amount",
        description="Amount of inset between extrusions (larger value = smaller inner face)",
        default=0.3,
        min=0.0,
        max=0.9,
        precision=3,
        subtype='FACTOR'
    )
    bpy.types.Scene.fractal_inset_depth = FloatProperty(
        name="Inset Depth",
        description="Depth of inset (0 for flat inset, negative for inward, positive for outward)",
        default=0.0,
        min=-0.5,
        max=0.5,
        precision=3
    )
    bpy.types.Scene.fractal_inset_relative = BoolProperty(
        name="Relative Inset",
        description="Scale inset by face size (creates more uniform insets across different face sizes)",
        default=True
    )
    bpy.types.Scene.fractal_inset_edges_only = BoolProperty(
        name="Edge Inset Only",
        description="Inset only affects edges, not moving faces along normals",
        default=False
    )
    bpy.types.Scene.fractal_second_extrude_factor = FloatProperty(
        name="Second Extrusion Factor",
        description="Factor for the second extrusion (relative to first extrusion)",
        default=0.7,
        min=0.1,
        max=2.0,
        precision=3,
        subtype='FACTOR'
    )
    bpy.types.Scene.fractal_first_extrude_amount = FloatProperty(
        name="First Extrusion Amount",
        description="Base amount for first extrusion before fractal modulation",
        default=0.5,
        min=0.1,
        max=2.0,
        precision=3
    )
    
    bpy.types.Scene.use_smooth_shading = BoolProperty(
        name="Smooth Shading",
        description="Apply smooth shading to the result",
        default=False
    )
    bpy.types.Scene.fractal_selected_only = BoolProperty(
        name="Selected Faces Only",
        description="Apply fractal only to selected faces",
        default=True
    )
    
    # Safety properties
    bpy.types.Scene.fractal_face_limit = IntProperty(
        name="Face Limit",
        description="Maximum number of faces to process",
        default=500,
        min=1,
        max=MAX_ALLOWED_FACES
    )
    bpy.types.Scene.fractal_batch_processing = BoolProperty(
        name="Batch Processing",
        description="Process faces in batches for better performance and UI responsiveness",
        default=True
    )
    bpy.types.Scene.fractal_batch_size = IntProperty(
        name="Batch Size",
        description="Number of faces to process in each batch",
        default=DEFAULT_BATCH_SIZE,
        min=10,
        max=1000
    )

def unregister_properties():
    # Remove window manager properties
    del bpy.types.WindowManager.fractal_is_processing
    del bpy.types.WindowManager.fractal_should_cancel
    del bpy.types.WindowManager.fractal_progress
    
    # Remove scene properties
    del bpy.types.Scene.fractal_iterations
    del bpy.types.Scene.fractal_scale
    del bpy.types.Scene.fractal_complexity
    del bpy.types.Scene.fractal_seed
    del bpy.types.Scene.fractal_type
    del bpy.types.Scene.fractal_pattern_type
    del bpy.types.Scene.fractal_inset_amount
    del bpy.types.Scene.fractal_inset_depth
    del bpy.types.Scene.fractal_second_extrude_factor
    del bpy.types.Scene.use_smooth_shading
    del bpy.types.Scene.fractal_selected_only
    del bpy.types.Scene.fractal_face_limit
    del bpy.types.Scene.fractal_batch_processing
    del bpy.types.Scene.fractal_batch_size

classes = (
    FRACTAL_PT_main_panel,
    MESH_OT_fractal_generate,
    MESH_OT_fractal_randomize_seed,
    MESH_OT_fractal_cancel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()

def unregister():
    unregister_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
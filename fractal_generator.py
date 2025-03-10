bl_info = {
    "name": "Advanced Fractal Geometry Generator",
    "author": "Your Name",
    "version": (1, 3),
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
import cmath
from mathutils import Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    FloatVectorProperty,
    EnumProperty,
    BoolProperty
)

# Helper functions

def safe_value(value, default, min_val=None, max_val=None):
    """Safely clamp a value within bounds to prevent instability"""
    if value is None or math.isnan(value) or math.isinf(value):
        return default
        
    result = value
    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)
    return result

def check_system_resources():
    """Check if system has enough resources to continue processing - with graceful fallback"""
    # Try to use psutil if available, but continue without it if not installed
    try:
        import psutil
        
        # Check available memory (at least 500MB free)
        try:
            mem = psutil.virtual_memory()
            if mem.available < 500 * 1024 * 1024:  # 500MB in bytes
                return False, "Low memory"
        except Exception as e:
            debug_print(f"Memory check error: {e}")
            pass  # Continue even if memory check fails
            
        return True, ""
        
    except ImportError:
        debug_print("psutil module not available - skipping resource check")
        # If psutil is not available, just return safe to continue
        return True, ""

# Global constants for safety
MAX_ALLOWED_FACES = 10000  # Prevents processing too many faces at once
DEFAULT_BATCH_SIZE = 500  # Process faces in batches to avoid memory spikes
DEBUG = False  # Set to True for additional console output
MAX_SAFE_VERTICES = 1000000  # Maximum vertex count to prevent crashes

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
            debug_print(f"Limiting selection from {len(selected_faces)} to {max_faces}")
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
        
        # Reset to Defaults Button
        row = layout.row(align=True)
        row.operator("mesh.fractal_reset_defaults", text="Reset to Defaults", icon='LOOP_BACK')

        # Basic Settings
        box = layout.box()
        box.label(text="Basic Settings", icon='SETTINGS')
        box.prop(scene, "fractal_type")
        box.prop(scene, "fractal_scale")
        box.prop(scene, "fractal_complexity")
        box.prop(scene, "fractal_selected_only")
        box.prop(scene, "fractal_face_limit")
        
        # Fractal-specific settings
        if scene.fractal_type in ['JULIA', 'JULIA_CUBIC', 'JULIA_QUARTIC', 'JULIA_QUINTIC']:
            box = layout.box()
            box.label(text="Julia Set Parameters", icon='FORCE_HARMONIC')
            box.prop(scene, "fractal_julia_seed_real")
            box.prop(scene, "fractal_julia_seed_imag")
            
            # Show the exponent control based on Julia set type
            if scene.fractal_type == 'JULIA':
                box.prop(scene, "fractal_power", text="Power (Z^n)")
            
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
        box.prop(scene, "use_symmetry")
        
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
        
        # Also randomize Julia set seed if that fractal type is selected
        if context.scene.fractal_type in ['JULIA', 'JULIA_CUBIC', 'JULIA_QUARTIC', 'JULIA_QUINTIC']:
            # Generate interesting Julia set parameters
            # Values near the edge of the Mandelbrot set often produce interesting Julia sets
            real = random.uniform(-1.0, 1.0)
            imag = random.uniform(-1.0, 1.0)
            
            # Make sure it's not too far outside the Mandelbrot set
            # (values too far outside tend to produce less interesting Julia sets)
            while real*real + imag*imag > 2.0:
                real = random.uniform(-1.0, 1.0)
                imag = random.uniform(-1.0, 1.0)
                
            context.scene.fractal_julia_seed_real = real
            context.scene.fractal_julia_seed_imag = imag
        
        self.report({'INFO'}, f"New random seed: {new_seed}")
        return {'FINISHED'}

class MESH_OT_fractal_reset_defaults(bpy.types.Operator):
    bl_idname = "mesh.fractal_reset_defaults"
    bl_label = "Reset to Defaults"
    bl_description = "Reset all fractal generator settings to their default values"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Reset basic properties
        scene.fractal_iterations = 50
        scene.fractal_scale = 1.5
        scene.fractal_complexity = 0.5
        scene.fractal_seed = 1
        scene.fractal_type = 'MANDELBROT'
        
        # Reset Julia set properties
        scene.fractal_julia_seed_real = -0.8
        scene.fractal_julia_seed_imag = 0.156
        scene.fractal_power = 2
        
        # Reset stepping pattern properties
        scene.fractal_extrude_along_normal = True
        scene.fractal_use_individual_normals = True
        scene.fractal_inset_amount = 0.3
        scene.fractal_inset_depth = 0.0
        scene.fractal_inset_relative = True
        scene.fractal_inset_edges_only = False
        scene.fractal_second_extrude_factor = 0.7
        scene.fractal_first_extrude_amount = 0.5
        
        # Reset other properties
        scene.use_smooth_shading = False
        scene.use_symmetry = True
        scene.fractal_selected_only = True
        scene.fractal_face_limit = 500
        scene.fractal_batch_processing = True
        scene.fractal_batch_size = DEFAULT_BATCH_SIZE
        
        self.report({'INFO'}, "Fractal settings reset to defaults")
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

    def check_operation_safety(self):
        """Check if the operation is safe to continue"""
        # Check if we've been processing too long
        if time.time() - self.start_time > 120:  # 2 minute limit
            self.report({'WARNING'}, "Operation taking too long - automatically cancelled")
            return False
        
        # Check if we have too many vertices
        if self.bm and len(self.bm.verts) > MAX_SAFE_VERTICES:
            self.report({'WARNING'}, f"Too many vertices created ({len(self.bm.verts)}), stopping for safety")
            return False
            
        # Check memory usage if psutil is available
        try:
            is_safe, reason = check_system_resources()
            if not is_safe:
                self.report({'WARNING'}, f"Operation cancelled: {reason}")
                return False
        except Exception as e:
            # If resource check itself fails, log but continue
            debug_print(f"Resource check error: {e}")
            
        return True

    def calculate_safe_iterations(self, scale):
        """Calculate safe iteration count based on zoom scale"""
        # More iterations needed at deeper zoom levels
        base_iterations = 50
        zoom_factor = math.log(1/scale) if scale > 0 else 0
        return min(base_iterations + int(zoom_factor * 20), 1000)
    
    def apply_symmetry(self, fractal_type, power=2):
        """Determine symmetry properties for different fractal types"""
        if fractal_type == 'MANDELBROT':
            # Mandelbrot set has conjugate symmetry (mirror across x-axis)
            return {'symmetry_type': 'MIRROR_X', 'fold': 2}
        elif fractal_type.startswith('JULIA'):
            # Julia sets have n-fold rotational symmetry where n is the power
            return {'symmetry_type': 'ROTATIONAL', 'fold': power}
        else:
            # Default case
            return {'symmetry_type': 'NONE', 'fold': 1}

    def process_complex_pattern(self, face, fractal_value, scene):
        """Process a face with extrude → insert → extrude pattern - with enhanced safety"""
        try:
            # Skip if face is not valid
            if not validate_face(face):
                return False
                
            # Skip faces with very low fractal values
            if fractal_value < 0.1:
                return False
                
            # Calculate face area and skip extremely small faces
            face_area = face.calc_area()
            if face_area < 0.00001:
                return False
                
            # Skip faces with too many vertices
            if len(face.verts) > 50:  # Reduced from 100 for better stability
                return False
                
            face_normal = face.normal.copy()
            if face_normal.length < 0.001:
                # Use global Z if normal is invalid
                face_normal = Vector((0, 0, 1))
            else:
                face_normal.normalize()
                
            # --- STEP 1: FIRST EXTRUSION with safety limits ---
            # Extrude the face
            result = bmesh.ops.extrude_face_region(self.bm, geom=[face])
            
            # Get newly created geometry
            new_geom = result.get("geom", [])
            if not new_geom:  # Safety check
                return False
                
            new_faces = [g for g in new_geom if isinstance(g, bmesh.types.BMFace)]
            new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]
            
            if not new_faces or not new_verts:  # Safety check
                return False
                
            # Calculate first extrusion strength with additional safety
            extrude_strength = scene.fractal_first_extrude_amount * fractal_value * scene.fractal_complexity
            extrude_strength = safe_value(extrude_strength, 0.1, 0.01, 1.0)  # More conservative cap
            
            # Generate a deterministic random factor with safety
            if scene.fractal_complexity > 0.5:
                try:
                    face_center = face.calc_center_median()
                    position_hash = (face_center.x * 1000 + face_center.y * 100 + face_center.z * 10) * scene.fractal_seed
                    # Generate a pseudo-random factor with tighter bounds
                    rand_factor = 0.9 + ((position_hash % 200) / 1000)  # Range: 0.9 to 1.1
                    extrude_strength *= rand_factor
                except:
                    # Skip randomization if calculation fails
                    pass
            
            # Move vertices along normal for first extrusion
            if new_verts:
                if scene.fractal_extrude_along_normal:
                    extrude_vec = face_normal * extrude_strength
                else:
                    extrude_vec = Vector((0, 0, extrude_strength))
                    
                # Safety check for extrusion vector
                if extrude_vec.length > 0.0001 and extrude_vec.length < 10.0:
                    bmesh.ops.translate(
                        self.bm,
                        vec=extrude_vec,
                        verts=new_verts
                    )
                
            # --- STEP 2: INSET FACES with safety limits---
            # Set up inset parameters with additional safety
            inset_amount = scene.fractal_inset_amount * fractal_value
            inset_amount = safe_value(inset_amount, 0.3, 0.01, 0.7)  # More conservative cap
            
            inset_depth = scene.fractal_inset_depth * fractal_value
            inset_depth = safe_value(inset_depth, 0.0, -0.3, 0.3)  # More conservative cap
            
            # Safety check for faces before inset
            valid_faces = [f for f in new_faces if validate_face(f)]
            if not valid_faces:
                return False
            
            # Inset with try/except for added safety
            try:
                inset_result = bmesh.ops.inset_region(
                    self.bm,
                    faces=valid_faces,
                    thickness=inset_amount,
                    depth=inset_depth,
                    use_even_offset=True,
                    use_interpolate=True,
                    use_relative_offset=scene.fractal_inset_relative,
                    use_edge_rail=scene.fractal_inset_edges_only
                )
                
                # Get the inner faces from the inset operation
                inner_faces = inset_result.get("faces", [])
            except Exception:
                # If inset fails, return what we've done so far
                return True
            
            # --- STEP 3: SECOND EXTRUSION with safety limits---
            # Only proceed if we have valid inner faces
            if inner_faces:
                # Limit the number of inner faces to process for stability
                max_inner_faces = min(len(inner_faces), 20)
                for i in range(max_inner_faces):
                    inner_face = inner_faces[i]
                    if not validate_face(inner_face):
                        continue
                    
                    try:
                        # Determine which normal to use
                        if scene.fractal_use_individual_normals:
                            inner_normal = inner_face.normal.copy()
                            if inner_normal.length < 0.001:
                                inner_normal = face_normal
                            else:
                                inner_normal.normalize()
                        else:
                            inner_normal = face_normal
                            
                        # Extrude this inner face (the second extrusion in our pattern)
                        face_extrude_result = bmesh.ops.extrude_face_region(self.bm, geom=[inner_face])
                        face_verts = [g for g in face_extrude_result.get("geom", []) 
                                      if isinstance(g, bmesh.types.BMVert)]
                        
                        if face_verts:
                            # The second extrusion with safety limits
                            second_extrude_strength = extrude_strength * scene.fractal_second_extrude_factor
                            second_extrude_strength = safe_value(second_extrude_strength, 0.1, 0.01, 0.8)
                            
                            if scene.fractal_extrude_along_normal:
                                second_extrude_vec = inner_normal * second_extrude_strength
                            else:
                                second_extrude_vec = Vector((0, 0, second_extrude_strength))
                            
                            # Safety check for extrusion vector
                            if (second_extrude_vec.length > 0.0001 and 
                                second_extrude_vec.length < 10.0):
                                bmesh.ops.translate(
                                    self.bm,
                                    vec=second_extrude_vec,
                                    verts=face_verts
                                )
                    except Exception:
                        # Skip this face if there's an error
                        continue
            
            return True
        
        except Exception as e:
            debug_print(f"Error in fractal stepping pattern: {str(e)}")
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
                # Get seed from properties
                seed_real = scene.fractal_julia_seed_real
                seed_imag = scene.fractal_julia_seed_imag
                power = scene.fractal_power
                return self.julia_value(x, y, iterations, seed_real, seed_imag, power)
            elif fractal_type == 'JULIA_CUBIC':
                seed_real = scene.fractal_julia_seed_real
                seed_imag = scene.fractal_julia_seed_imag
                return self.julia_value(x, y, iterations, seed_real, seed_imag, 3)
            elif fractal_type == 'JULIA_QUARTIC':
                seed_real = scene.fractal_julia_seed_real
                seed_imag = scene.fractal_julia_seed_imag
                return self.julia_value(x, y, iterations, seed_real, seed_imag, 4)
            elif fractal_type == 'JULIA_QUINTIC':
                seed_real = scene.fractal_julia_seed_real
                seed_imag = scene.fractal_julia_seed_imag
                return self.julia_value(x, y, iterations, seed_real, seed_imag, 5)
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
        """Calculate Mandelbrot set value with enhanced optimization and safety checks"""
        try:
            # Limit input values to prevent extreme calculations
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            max_iter = min(max_iter, 500)  # Hard cap on iterations
            
            # Early bailout optimization
            # Check if point is in main cardioid or period-2 bulb
            q = (x - 0.25)**2 + y**2
            if q * (q + (x - 0.25)) < 0.25 * (y**2):
                return 1.0  # Inside cardioid
            
            if (x + 1.0)**2 + y**2 < 0.0625:
                return 1.0  # Inside period-2 bulb
                
            # Main iteration with smooth coloring
            c = complex(x, y)
            z = complex(0, 0)
            
            # Optimized escape detection
            for i in range(max_iter):
                # Using z_squared directly saves one complex multiplication per iteration
                z_squared = complex(z.real * z.real - z.imag * z.imag, 2 * z.real * z.imag)
                z = z_squared + c
                
                # Check escape condition using squared magnitude
                mag_squared = z.real * z.real + z.imag * z.imag
                
                if mag_squared > 4.0:  # Radius 2 squared = 4
                    # Smooth coloring formula for better visual results
                    smooth_i = i + 1 - math.log(math.log(mag_squared)) / math.log(2)
                    return smooth_i / max_iter
                
                # Check for numerical instability
                if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                    return 0.0
            
            # If we reached max iterations, point is in the set
            return 1.0
        except Exception as e:
            # Fallback value if calculation fails
            debug_print(f"Mandelbrot calculation error: {e}")
            return 0.3
    
    def julia_value(self, x, y, max_iter, seed_real=-0.8, seed_imag=0.156, power=2):
        """Calculate Julia set value with custom seed and power"""
        try:
            # Limit input values to prevent extreme calculations
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            max_iter = min(max_iter, 500)
            
            # The c parameter is fixed for Julia sets
            c = complex(seed_real, seed_imag)
            
            # Starting z is the input coordinate
            z = complex(x, y)
            
            # Main iteration loop with smooth coloring
            for i in range(max_iter):
                # Special case for power=2 (most common)
                if power == 2:
                    z = z * z + c
                else:
                    # For other powers, use cmath.pow
                    z = cmath.pow(z, power) + c
                
                # Check escape condition
                mag_squared = z.real * z.real + z.imag * z.imag
                
                if mag_squared > 4.0:
                    # Smooth coloring
                    smooth_i = i + 1 - math.log(math.log(mag_squared)) / math.log(2)
                    return smooth_i / max_iter
                
                # Check for numerical instability
                if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                    return 0.0
            
            # If we reached max iterations, point is in the set
            return 1.0
        except Exception as e:
            debug_print(f"Julia calculation error: {e}")
            return 0.3
    
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
                # Simple smooth coloring for 3D
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
        """Modal handler for batch processing with enhanced safety"""
        try:
            if event.type == 'ESC' or context.window_manager.fractal_should_cancel:
                self.report({'INFO'}, "Fractal generation cancelled")
                self.cleanup(context, cancelled=True)
                return {'CANCELLED'}
                
            if event.type == 'TIMER':
                # Safety check
                if not self.check_operation_safety():
                    self.cleanup(context, cancelled=True)
                    return {'CANCELLED'}
                    
                # Update progress
                if hasattr(context.window_manager, 'fractal_progress'):
                    progress = (self.current_batch / len(self.face_batches)) * 100
                    context.window_manager.fractal_progress = progress
                
                # Process next batch
                if self.current_batch < len(self.face_batches):
                    success = self.process_batch(context, self.current_batch)
                    if not success:
                        self.report({'WARNING'}, "Batch processing failed, stopping safely")
                        self.cleanup(context, cancelled=True)
                        return {'CANCELLED'}
                        
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
                
                # Check for symmetry optimizations
                if scene.use_symmetry:
                    # Get the symmetry properties based on fractal type
                    if scene.fractal_type == 'JULIA':
                        symmetry = self.apply_symmetry(scene.fractal_type, scene.fractal_power)
                    else:
                        symmetry = self.apply_symmetry(scene.fractal_type)
                        
                    # Use symmetry to optimize calculations
                    if symmetry['symmetry_type'] == 'MIRROR_X' and center.y < 0:
                        # For Mandelbrot with y-axis mirror symmetry, reflect point across x-axis
                        mirror_center = Vector((center.x, -center.y, center.z))
                        fractal_value = self.get_fractal_value(
                            mirror_center, scene.fractal_type, scene
                        )
                    else:
                        # Calculate normally for other symmetry types or original quadrant
                        fractal_value = self.get_fractal_value(
                            center, scene.fractal_type, scene
                        )
                else:
                    # No symmetry optimization, calculate normally
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
                
                # Safety check for vertex count during processing
                if len(self.bm.verts) > MAX_SAFE_VERTICES:
                    self.report({'WARNING'}, f"Vertex limit reached ({len(self.bm.verts)}), stopping batch")
                    break
            
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
        max=500
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
            ('JULIA', "Julia", "Julia set with custom power"),
            ('JULIA_CUBIC', "Julia Cubic", "Julia set with power=3"),
            ('JULIA_QUARTIC', "Julia Quartic", "Julia set with power=4"),
            ('JULIA_QUINTIC', "Julia Quintic", "Julia set with power=5"),
            ('QUINTIC_MANDELBULB', "Quintic Mandelbulb", "3D Quintic Mandelbulb (power=5)"),
            ('CUBIC_MANDELBULB', "Cubic Mandelbulb", "3D Cubic Mandelbulb (power=3)"),
        ],
        default='MANDELBROT'
    )
    
    # Julia set specific properties
    bpy.types.Scene.fractal_julia_seed_real = FloatProperty(
        name="Julia Seed Real",
        description="Real part of the complex constant C for Julia sets",
        default=-0.8,
        min=-2.0,
        max=2.0,
        precision=4
    )
    bpy.types.Scene.fractal_julia_seed_imag = FloatProperty(
        name="Julia Seed Imaginary",
        description="Imaginary part of the complex constant C for Julia sets",
        default=0.156,
        min=-2.0,
        max=2.0,
        precision=4
    )
    bpy.types.Scene.fractal_power = IntProperty(
        name="Power",
        description="Exponent power for Julia sets (z^n + c)",
        default=2,
        min=2,
        max=9
    )
    
    # Symmetry properties
    bpy.types.Scene.use_symmetry = BoolProperty(
        name="Use Symmetry",
        description="Optimize calculation using fractal symmetry properties",
        default=True
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
    """Unregister properties associated with the fractal generator."""
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
    del bpy.types.Scene.use_smooth_shading
    del bpy.types.Scene.fractal_selected_only
    del bpy.types.Scene.fractal_face_limit
    del bpy.types.Scene.fractal_batch_processing
    del bpy.types.Scene.fractal_batch_size
    
    # Remove Julia-specific properties
    del bpy.types.Scene.fractal_julia_seed_real
    del bpy.types.Scene.fractal_julia_seed_imag
    del bpy.types.Scene.fractal_power
    
    # Remove symmetry properties
    del bpy.types.Scene.use_symmetry
    
    # Remove pattern-specific properties
    del bpy.types.Scene.fractal_inset_amount
    del bpy.types.Scene.fractal_inset_depth
    del bpy.types.Scene.fractal_inset_relative
    del bpy.types.Scene.fractal_inset_edges_only
    del bpy.types.Scene.fractal_second_extrude_factor
    del bpy.types.Scene.fractal_first_extrude_amount
    del bpy.types.Scene.fractal_extrude_along_normal
    del bpy.types.Scene.fractal_use_individual_normals

classes = (
    FRACTAL_PT_main_panel,
    MESH_OT_fractal_generate,
    MESH_OT_fractal_randomize_seed,
    MESH_OT_fractal_reset_defaults,
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
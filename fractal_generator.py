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
import functools
from mathutils import Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    FloatVectorProperty,
    EnumProperty,
    BoolProperty
)

# Try to import NumPy for optimized calculations - with graceful fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

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

# Value caching system for fractal calculations
FRACTAL_CACHE_SIZE = 1000  # Maximum number of cached fractal values
FRACTAL_CACHE = {}  # Global cache dictionary
FRACTAL_CACHE_HITS = 0  # For diagnostics
FRACTAL_CACHE_PRECISION = 3  # Decimal precision for cache keys

# Constants for fractal calculation optimization
MANDELBROT_CARDIOID_PERIOD2_OPTIMIZATION = True  # Early bailout for main cardioid and period-2 bulb
ADAPTIVE_ITERATIONS = True  # Use variable iteration count based on zoom level
MAX_BATCH_PROCESSING_TIME = 0.5  # Maximum time (seconds) per batch to keep UI responsive

# Visual coherence and stability settings
ENABLE_VALUE_NORMALIZATION = True  # Apply non-linear normalization to fractal values
ENABLE_PATTERN_COHERENCE = True  # Apply smoothing between neighboring faces
ENABLE_EXTRUSION_CONTROL = True  # Use controlled mapping for extrusion parameters
NEIGHBOR_INFLUENCE = 0.3  # How much neighboring faces influence each other (0.0-1.0)
MAX_COHERENCE_NEIGHBORS = 5  # Maximum number of neighbors to consider for coherence

def debug_print(message):
    """Print debug messages to console if DEBUG is enabled"""
    if DEBUG:
        print(f"[Fractal Generator] {message}")

def cached_fractal_calc(func):
    """Function decorator for caching fractal calculations to improve performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global FRACTAL_CACHE, FRACTAL_CACHE_HITS
        
        # Create a cache key based on function name and arguments
        # Round spatial coordinates to reduce cache variations
        rounded_args = []
        
        for arg in args:
            if isinstance(arg, (int, float)):
                rounded_args.append(round(arg, FRACTAL_CACHE_PRECISION))
            else:
                rounded_args.append(arg)
                
        cache_key = (func.__name__, tuple(rounded_args))
        
        # Check if we have this result cached
        if cache_key in FRACTAL_CACHE:
            FRACTAL_CACHE_HITS += 1
            return FRACTAL_CACHE[cache_key]
            
        # Calculate the new value
        result = func(*args, **kwargs)
        
        # Cache the result if we haven't exceeded cache size
        if len(FRACTAL_CACHE) < FRACTAL_CACHE_SIZE:
            FRACTAL_CACHE[cache_key] = result
        elif len(FRACTAL_CACHE) >= FRACTAL_CACHE_SIZE and random.random() < 0.1:
            # Randomly replace entries once cache is full to avoid staleness
            keys = list(FRACTAL_CACHE.keys())
            if keys:
                del FRACTAL_CACHE[random.choice(keys)]
                FRACTAL_CACHE[cache_key] = result
                
        return result
    
    return wrapper

def normalize_fractal_value(raw_value, fractal_type):
    """Normalize and smooth fractal values for more predictable geometry"""
    # Only apply normalization if enabled
    if not ENABLE_VALUE_NORMALIZATION:
        return max(0.0, min(1.0, raw_value))
        
    # Base normalization to ensure we're in [0,1] range
    value = max(0.0, min(1.0, raw_value))
    
    # Apply non-linear scaling for more visual coherence
    if fractal_type.startswith('JULIA'):
        # Julia sets need smoother transition
        return math.pow(value, 1.5)  # Emphasize higher values
    elif fractal_type.endswith('MANDELBULB'):
        # 3D Mandelbulbs need stronger thresholding
        threshold = 0.2
        if value < threshold:
            return 0.0
        return ((value - threshold) / (1.0 - threshold)) ** 1.2
    else:  # Mandelbrot
        # Apply sigmoid-like smoothing for Mandelbrot
        if value < 0.3:
            return value * 0.5  # Reduce low values
        elif value > 0.7:
            return 0.7 + (value - 0.7) * 0.7  # Gentle compression of high values
        return value

def calculate_extrusion_params(face, fractal_value, scene):
    """Calculate extrusion parameters with improved consistency"""
    # Skip controlled mapping if disabled
    if not ENABLE_EXTRUSION_CONTROL:
        return {
            'first_extrude': scene.fractal_first_extrude_amount * fractal_value,
            'inset_amount': scene.fractal_inset_amount,
            'inset_depth': scene.fractal_inset_depth * fractal_value,
            'second_extrude': scene.fractal_first_extrude_amount * fractal_value * scene.fractal_second_extrude_factor
        }
    
    # Normalize the fractal value first
    norm_value = normalize_fractal_value(fractal_value, scene.fractal_type)
    
    # Scale factor based on face size for more uniform results
    face_area = face.calc_area()
    area_factor = min(1.0, max(0.3, math.sqrt(face_area) * 2.0))
    
    # Apply controlled mapping for first extrusion
    if norm_value < 0.2:
        # Minimal extrusion for low values
        first_extrude = scene.fractal_first_extrude_amount * 0.2 * area_factor
    elif norm_value > 0.8:
        # Cap maximum extrusion for high values
        first_extrude = scene.fractal_first_extrude_amount * 0.8 * area_factor
    else:
        # Linear mapping for middle range
        first_extrude = scene.fractal_first_extrude_amount * norm_value * area_factor
    
    # Similar controlled approach for inset
    inset_amount = scene.fractal_inset_amount * (0.5 + norm_value * 0.5) * 0.8
    
    # Second extrusion uses more dampened value to prevent extreme variations
    second_extrude = first_extrude * scene.fractal_second_extrude_factor * 0.7
    
    return {
        'first_extrude': first_extrude,
        'inset_amount': inset_amount,
        'inset_depth': scene.fractal_inset_depth * norm_value * 0.5,  # Reduced impact
        'second_extrude': second_extrude
    }

def select_coherent_faces(bm, selected_faces, max_faces=500):
    """Select faces with better spatial coherence for more predictable results"""
    if not selected_faces or not ENABLE_PATTERN_COHERENCE:
        return selected_faces
    
    # Start with seed faces from selection
    result_faces = selected_faces[:min(10, len(selected_faces))]
    candidate_faces = set(selected_faces) - set(result_faces)
    
    # Build coherent groups by proximity
    while len(result_faces) < max_faces and candidate_faces:
        # Find the closest candidate to any face in our result set
        best_candidate = None
        best_distance = float('inf')
        
        for face in list(candidate_faces)[:100]:  # Limit search for performance
            face_center = face.calc_center_median()
            
            # Find closest distance to any already-selected face
            for selected in result_faces[-10:]:  # Check against recent selections
                selected_center = selected.calc_center_median()
                dist = (face_center - selected_center).length
                
                if dist < best_distance:
                    best_distance = dist
                    best_candidate = face
        
        if best_candidate:
            result_faces.append(best_candidate)
            candidate_faces.remove(best_candidate)
        else:
            break
    
    return result_faces

def apply_coherence_pattern(faces, face_values, fractal_type):
    """Apply smoothing to ensure neighboring faces have similar fractal values"""
    if len(faces) <= 1 or not ENABLE_PATTERN_COHERENCE:
        return face_values
    
    # Create a mapping of faces to their values
    face_to_value = {face: value for face, value in zip(faces, face_values)}
    
    # Find adjacent faces for each face
    for face in faces:
        adjacent_faces = set()
        for edge in face.edges:
            for adjacent_face in edge.link_faces:
                if adjacent_face in face_to_value and adjacent_face != face:
                    adjacent_faces.add(adjacent_face)
        
        # Skip if no adjacent faces
        if not adjacent_faces:
            continue
        
        # Limit to closest neighbors to prevent over-smoothing
        if len(adjacent_faces) > MAX_COHERENCE_NEIGHBORS:
            # Convert to list for sorting
            adjacent_list = list(adjacent_faces)
            # Sort by distance to current face center
            face_center = face.calc_center_median()
            adjacent_list.sort(key=lambda adj: (adj.calc_center_median() - face_center).length)
            # Keep only closest neighbors
            adjacent_faces = set(adjacent_list[:MAX_COHERENCE_NEIGHBORS])
        
        # Apply neighborhood smoothing (average with neighbors)
        adjacent_values = [face_to_value[adj_face] for adj_face in adjacent_faces]
        avg_value = sum(adjacent_values) / len(adjacent_values)
        
        # Blend original value with neighborhood average
        influence = NEIGHBOR_INFLUENCE
        # Special case for Mandelbulbs - use less influence to preserve detail
        if fractal_type.endswith('MANDELBULB'):
            influence *= 0.7
            
        face_to_value[face] = face_to_value[face] * (1.0 - influence) + avg_value * influence
    
    # Return smoothed values
    return [face_to_value[face] for face in faces]

def align_extrusion_normal(face, face_normal, neighboring_faces):
    """Adjust extrusion direction for better visual coherence with neighbors"""
    if not neighboring_faces or not ENABLE_PATTERN_COHERENCE:
        return face_normal
    
    # Get average normal of neighboring faces
    avg_normal = Vector((0, 0, 0))
    for neighbor in neighboring_faces:
        if neighbor.normal.length > 0.0001:
            avg_normal += neighbor.normal
    
    if avg_normal.length < 0.0001:
        return face_normal
    
    avg_normal.normalize()
    
    # Blend face normal with neighborhood normal
    # More weight to face normal to preserve detail
    blend_factor = min(0.8, 1.0 - NEIGHBOR_INFLUENCE)  # At least 20% face normal
    blended_normal = face_normal * blend_factor + avg_normal * (1.0 - blend_factor)
    
    if blended_normal.length < 0.0001:
        return face_normal
        
    blended_normal.normalize()
    return blended_normal

def stable_mandelbulb_iteration(x, y, z, power):
    """More stable power function for 3D Mandelbulbs"""
    # Convert to spherical
    r = math.sqrt(x*x + y*y + z*z)
    
    # Handle near-zero case more gracefully
    if r < 0.000001:
        return x, y, z
    
    # More stable spherical conversion
    theta = math.atan2(math.sqrt(x*x + y*y), z)
    phi = math.atan2(y, x)
    
    # Control power to avoid extreme values
    actual_power = min(power, 8.0)  # Cap maximum power
    
    # Apply power function with stability controls
    r_pow = min(pow(r, actual_power), 1000.0)
    
    # Prevent oscillation at high values
    if r_pow > 100.0:
        damping = 100.0 / r_pow
        r_pow = 100.0
    else:
        damping = 1.0
    
    # Apply spherical functions with damping
    sin_theta = math.sin(theta * actual_power) * damping
    cos_theta = math.cos(theta * actual_power) * damping
    sin_phi = math.sin(phi * actual_power) * damping
    cos_phi = math.cos(phi * actual_power) * damping
    
    # Convert back to cartesian with controlled magnitude
    new_x = r_pow * sin_theta * cos_phi
    new_y = r_pow * sin_theta * sin_phi
    new_z = r_pow * cos_theta
    
    return new_x, new_y, new_z

def validate_face(face):
    """Check if a face is valid for processing with enhanced stability checks"""
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
        
        # Additional checks for face stability
        # Check for extremely acute angles - these can cause numerical issues
        for edge in face.edges:
            if len(edge.link_faces) < 2:
                continue  # Skip boundary edges
                
            # Calculate angle between face normals - reject faces with very sharp edges
            # that would create instability in subdivision operations
            face1, face2 = edge.link_faces
            if face1 and face2:
                try:
                    angle = face1.normal.angle(face2.normal)
                    # Reject faces with very acute or very obtuse angles
                    if angle > 2.8:  # Close to pi (180 degrees)
                        return False
                except:
                    pass
        
        # Check aspect ratio - extremely long thin faces can cause issues
        try:
            # Simple aspect ratio check - get rough bounds
            verts = [v.co for v in face.verts]
            min_x = min(v.x for v in verts)
            max_x = max(v.x for v in verts)
            min_y = min(v.y for v in verts)
            max_y = max(v.y for v in verts)
            min_z = min(v.z for v in verts)
            max_z = max(v.z for v in verts)
            
            width = max(max_x - min_x, max_y - min_y, max_z - min_z)
            height = min(max_x - min_x, max_y - min_y, max_z - min_z)
            if height > 0 and width / height > 50:  # Very extreme aspect ratio
                return False
        except:
            pass
            
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
        
        # Visual Coherence Settings
        box = layout.box()
        box.label(text="Visual Coherence", icon='SHADERFX')
        box.prop(scene, "fractal_enable_value_normalization")
        box.prop(scene, "fractal_enable_coherence")
        if scene.fractal_enable_coherence:
            box.prop(scene, "fractal_coherence_strength")
        box.prop(scene, "fractal_enable_extrusion_control")
        
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
        
        # Reset coherence properties
        scene.fractal_enable_value_normalization = True
        scene.fractal_enable_coherence = True
        scene.fractal_coherence_strength = 0.3
        scene.fractal_enable_extrusion_control = True
        
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

    def calculate_safe_iterations(self, scale, x=0, y=0):
        """Calculate optimized iteration count based on zoom scale and position
        
        Adaptively chooses iteration count based on:
        1. Zoom level - deeper zooms need more iterations for detail
        2. Position in the fractal - areas near the boundary need more iterations
        3. Distance from origin - outer areas often escape quickly and need fewer iterations
        """
        if not ADAPTIVE_ITERATIONS:
            # Use fixed iteration count if adaptive mode is disabled
            return min(50, 1000)
            
        # Base iteration count
        base_iterations = 50
        
        # Scale factor - deeper zooms need more iterations
        zoom_factor = math.log(1/scale) if scale > 0 else 0
        iterations = base_iterations + int(zoom_factor * 20)
        
        # Position factor - adjust based on distance from origin
        # Points further from origin generally need fewer iterations
        distance_from_origin = math.sqrt(x*x + y*y)
        
        # Reduce iterations for far points, but ensure minimum count
        if distance_from_origin > 1.5:
            iterations = max(20, int(iterations * (2.0 / distance_from_origin)))
        
        # Points very close to origin often need fewer iterations too
        # (inside main cardioid/period-2 bulb for Mandelbrot)
        if distance_from_origin < 0.5:
            iterations = max(20, int(iterations * 0.7))
            
        # Cap iterations to prevent excessive calculation
        return min(iterations, 1000)
    
    def apply_symmetry(self, fractal_type, power=2):
        """Determine symmetry properties for different fractal types with enhanced handling"""
        if fractal_type == 'MANDELBROT':
            # Mandelbrot set has real axis symmetry (mirror across x-axis)
            # and additional 180° rotational symmetry around the origin
            return {
                'symmetry_type': 'MIRROR_X',
                'fold': 2,
                'has_additional_symmetry': True,
                'additional_type': 'ROTATIONAL',
                'additional_fold': 2
            }
        elif fractal_type.startswith('JULIA'):
            # Julia sets have n-fold rotational symmetry where n is the power
            power_factor = power
            
            # Special case for even powers (more symmetry)
            if power % 2 == 0:
                return {
                    'symmetry_type': 'ROTATIONAL',
                    'fold': power,
                    'has_additional_symmetry': True,
                    'additional_type': 'MIRROR_XY',
                    'additional_fold': 2
                }
            else:
                # Odd powers have pure rotational symmetry
                return {
                    'symmetry_type': 'ROTATIONAL',
                    'fold': power,
                    'has_additional_symmetry': False
                }
        elif fractal_type.endswith('MANDELBULB'):
            # 3D Mandelbulbs have spherical symmetry with power-fold rotational symmetry
            if 'QUINTIC' in fractal_type:
                power_val = 5
            elif 'CUBIC' in fractal_type:
                power_val = 3
            else:
                power_val = 2
                
            return {
                'symmetry_type': 'SPHERICAL',
                'fold': power_val,
                'has_additional_symmetry': True,
                'additional_type': 'ROTATIONAL',
                'additional_fold': power_val
            }
        else:
            # Default case - no symmetry
            return {
                'symmetry_type': 'NONE',
                'fold': 1,
                'has_additional_symmetry': False
            }

    def process_complex_pattern(self, face, fractal_value, scene, neighboring_faces=None):
        """Process a face with extrude → insert → extrude pattern - with enhanced coherence and stability"""
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
                
            # Get face normal with optional coherence alignment
            face_normal = face.normal.copy()
            if face_normal.length < 0.001:
                # Use global Z if normal is invalid
                face_normal = Vector((0, 0, 1))
            else:
                face_normal.normalize()
                
            # Apply normal alignment if coherence is enabled and we have neighboring faces
            if ENABLE_PATTERN_COHERENCE and neighboring_faces:
                face_normal = align_extrusion_normal(face, face_normal, neighboring_faces)
                
            # --- STEP 1: FIRST EXTRUSION with controlled parameters ---
            # Get extrusion parameters using improved coherence
            if ENABLE_EXTRUSION_CONTROL:
                params = calculate_extrusion_params(face, fractal_value, scene)
                extrude_strength = params['first_extrude']
                inset_amount = params['inset_amount']
                inset_depth = params['inset_depth']
                second_extrude_strength = params['second_extrude']
            else:
                # Legacy parameter calculation with safety limits
                extrude_strength = scene.fractal_first_extrude_amount * fractal_value * scene.fractal_complexity
                extrude_strength = safe_value(extrude_strength, 0.1, 0.01, 1.0)
                
                inset_amount = scene.fractal_inset_amount * fractal_value
                inset_amount = safe_value(inset_amount, 0.3, 0.01, 0.7)
                
                inset_depth = scene.fractal_inset_depth * fractal_value
                inset_depth = safe_value(inset_depth, 0.0, -0.3, 0.3)
                
                second_extrude_strength = extrude_strength * scene.fractal_second_extrude_factor
                second_extrude_strength = safe_value(second_extrude_strength, 0.1, 0.01, 0.8)
            
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
                
            # Generate a deterministic random factor with safety
            if scene.fractal_complexity > 0.5 and not ENABLE_EXTRUSION_CONTROL:
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
                
            # --- STEP 2: INSET FACES with controlled parameters ---
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
            
            # --- STEP 3: SECOND EXTRUSION with controlled parameters ---
            # Only proceed if we have valid inner faces
            if inner_faces:
                # Limit the number of inner faces to process for stability
                max_inner_faces = min(len(inner_faces), 20)
                for i in range(max_inner_faces):
                    inner_face = inner_faces[i]
                    if not validate_face(inner_face):
                        continue
                    
                    try:
                        # Determine which normal to use, with coherence if enabled
                        if scene.fractal_use_individual_normals:
                            inner_normal = inner_face.normal.copy()
                            if inner_normal.length < 0.001:
                                inner_normal = face_normal
                            else:
                                inner_normal.normalize()
                                
                            # Apply normal coherence if enabled
                            if ENABLE_PATTERN_COHERENCE and neighboring_faces:
                                inner_normal = align_extrusion_normal(inner_face, inner_normal, neighboring_faces)
                        else:
                            inner_normal = face_normal
                            
                        # Extrude this inner face (the second extrusion in our pattern)
                        face_extrude_result = bmesh.ops.extrude_face_region(self.bm, geom=[inner_face])
                        face_verts = [g for g in face_extrude_result.get("geom", []) 
                                      if isinstance(g, bmesh.types.BMVert)]
                        
                        if face_verts:
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
        """Calculate fractal value based on type with enhanced safety and optimizations"""
        try:
            # Cap scale for safety and better numerical stability
            scale = min(scene.fractal_scale, 10.0)
            
            # Use adaptive iteration count based on zoom level and position
            if ADAPTIVE_ITERATIONS:
                iterations = self.calculate_safe_iterations(scale, center.x, center.y)
            else:
                iterations = min(scene.fractal_iterations, 200)  # Cap iterations for safety
            
            # Apply scaling with improved numerical precision
            x = center.x * scale
            y = center.y * scale
            z = center.z * scale
            
            # Clear cache if it's grown too large to prevent memory issues
            global FRACTAL_CACHE, FRACTAL_CACHE_HITS
            if len(FRACTAL_CACHE) > FRACTAL_CACHE_SIZE * 1.5:
                debug_print(f"Clearing fractal cache (size: {len(FRACTAL_CACHE)}, hits: {FRACTAL_CACHE_HITS})")
                FRACTAL_CACHE.clear()
                FRACTAL_CACHE_HITS = 0
            
            # Call appropriate fractal calculation function with optimized dispatch
            raw_value = 0.5  # Default value
            
            if fractal_type == 'MANDELBROT':
                raw_value = self.mandelbrot_value(x, y, iterations)
                
            elif fractal_type.startswith('JULIA'):
                # Get seed parameters from properties
                seed_real = scene.fractal_julia_seed_real
                seed_imag = scene.fractal_julia_seed_imag
                
                # Dispatch to correct Julia function based on type with proper power value
                if fractal_type == 'JULIA':
                    power = scene.fractal_power
                elif fractal_type == 'JULIA_CUBIC':
                    power = 3
                elif fractal_type == 'JULIA_QUARTIC':
                    power = 4
                elif fractal_type == 'JULIA_QUINTIC':
                    power = 5
                else:
                    power = 2
                    
                raw_value = self.julia_value(x, y, iterations, seed_real, seed_imag, power)
                
            elif fractal_type == 'QUINTIC_MANDELBULB':
                raw_value = self.quintic_mandelbulb_value(x, y, z, iterations)
                
            elif fractal_type == 'CUBIC_MANDELBULB':
                raw_value = self.cubic_mandelbulb_value(x, y, z, iterations)
                
            # Apply normalization for improved visual coherence if enabled
            if ENABLE_VALUE_NORMALIZATION:
                return normalize_fractal_value(raw_value, fractal_type)
            else:
                return raw_value
                
        except Exception as e:
            # More informative error handling with traceback for debugging
            debug_print(f"Fractal calculation error: {e}")
            if DEBUG:
                traceback.print_exc()
            # Fallback value if calculation fails - avoid returning 0 to prevent dead spots
            return 0.3
    
    @cached_fractal_calc
    def mandelbrot_value(self, x, y, max_iter):
        """Calculate Mandelbrot set value with enhanced optimization and safety checks"""
        try:
            # Limit input values to prevent extreme calculations
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            max_iter = min(max_iter, 500)  # Hard cap on iterations
            
            # Early bailout optimization for main cardioid and period-2 bulb
            if MANDELBROT_CARDIOID_PERIOD2_OPTIMIZATION:
                # Check if point is in main cardioid
                q = (x - 0.25)**2 + y**2
                if q * (q + (x - 0.25)) < 0.25 * (y**2):
                    return 1.0  # Inside cardioid
                
                # Check if point is in period-2 bulb
                if (x + 1.0)**2 + y**2 < 0.0625:
                    return 1.0  # Inside period-2 bulb
            
            # Optimized calculation using NumPy if available
            if NUMPY_AVAILABLE:
                return self._mandelbrot_numpy(x, y, max_iter)
            else:
                return self._mandelbrot_standard(x, y, max_iter)
                
        except Exception as e:
            # Fallback value if calculation fails
            debug_print(f"Mandelbrot calculation error: {e}")
            return 0.3
            
    def _mandelbrot_standard(self, x, y, max_iter):
        """Standard Python implementation of Mandelbrot calculation"""
        c = complex(x, y)
        z = complex(0, 0)
        
        # Pre-calculate escape radius squared
        escape_radius_squared = 4.0
        
        # Track last z value for distance estimation if needed
        last_z = z
        
        # Optimized escape detection
        for i in range(max_iter):
            # Using z_squared directly saves one complex multiplication per iteration
            z_squared = complex(z.real * z.real - z.imag * z.imag, 2 * z.real * z.imag)
            last_z = z
            z = z_squared + c
            
            # Check escape condition using squared magnitude
            mag_squared = z.real * z.real + z.imag * z.imag
            
            if mag_squared > escape_radius_squared:
                # Smooth coloring formula for better visual results
                # Using logarithmic smoothing for better gradients near boundary
                smooth_i = i + 1 - math.log(math.log(mag_squared)) / math.log(2)
                return smooth_i / max_iter
            
            # Check for numerical instability
            if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                return 0.0
                
            # Periodicity checking - if we revisit a value, we're in a cycle
            # Skip this additional optimization for standard implementation as it adds overhead
        
        # If we reached max iterations, point is in the set
        return 1.0
        
    def _mandelbrot_numpy(self, x, y, max_iter):
        """NumPy-optimized implementation of Mandelbrot calculation"""
        # Create complex number from coordinates
        c = complex(x, y)
        
        # Use NumPy for vectorized operations
        # For single point calculation, this mainly helps with the smooth coloring
        
        # Initialize z as a complex number
        z = complex(0, 0)
        
        # Track iterations until escape
        for i in range(max_iter):
            # Optimized squaring and addition
            z = z*z + c
            
            # Check escape condition
            if abs(z) > 2.0:
                # Use numpy for smooth coloring calculation
                mag_squared = np.abs(z)**2
                smooth_i = i + 1 - np.log(np.log(mag_squared)) / np.log(2.0)
                return float(smooth_i / max_iter)
                
            # Safety check for numerical instability
            if np.isnan(z.real) or np.isnan(z.imag) or np.isinf(z.real) or np.isinf(z.imag):
                return 0.0
                
        # Point is in the set if we reached max iterations
        return 1.0
    
    @cached_fractal_calc
    def julia_value(self, x, y, max_iter, seed_real=-0.8, seed_imag=0.156, power=2):
        """Calculate Julia set value with custom seed and power using optimized methods"""
        try:
            # Limit input values to prevent extreme calculations
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            max_iter = min(max_iter, 500)
            
            # Check if parameter c is in the optimizable range
            # Some c values require more careful computation due to numerical sensitivity
            c_mag_squared = seed_real*seed_real + seed_imag*seed_imag
            requires_high_precision = c_mag_squared > 1.99 or power > 4
            
            # The c parameter is fixed for Julia sets
            c = complex(seed_real, seed_imag)
            
            # Choose the appropriate calculation method
            if NUMPY_AVAILABLE and not requires_high_precision:
                return self._julia_numpy(x, y, max_iter, c, power)
            else:
                return self._julia_standard(x, y, max_iter, c, power)
                
        except Exception as e:
            debug_print(f"Julia calculation error: {e}")
            return 0.3
            
    def _julia_standard(self, x, y, max_iter, c, power):
        """Standard Python implementation of Julia set calculation with enhanced stability"""
        # Starting z is the input coordinate
        z = complex(x, y)
        
        # Pre-calculate escape radius squared
        # For higher powers, we need a different escape radius
        if power > 2:
            # Higher powers need a larger escape radius for proper detection
            escape_radius_squared = max(4.0, power * 2.0)
        else:
            escape_radius_squared = 4.0
            
        # Main iteration loop with smooth coloring
        for i in range(max_iter):
            # Different power handling with optimizations
            if power == 2:
                # Most common case - optimize by manually squaring
                z = complex(z.real * z.real - z.imag * z.imag, 2 * z.real * z.imag) + c
            elif power == 3:
                # Cubic case - manual calculation to avoid cmath.pow overhead
                real = z.real*z.real*z.real - 3*z.real*z.imag*z.imag
                imag = 3*z.real*z.real*z.imag - z.imag*z.imag*z.imag
                z = complex(real, imag) + c
            elif power == 4:
                # Quartic case - manual calculation 
                z2 = complex(z.real * z.real - z.imag * z.imag, 2 * z.real * z.imag)
                z = complex(z2.real * z2.real - z2.imag * z2.imag, 2 * z2.real * z2.imag) + c
            elif power == 5:
                # Quintic case - special handling for stability
                # Using complex multiplication is more stable than direct formula for high powers
                z2 = complex(z.real * z.real - z.imag * z.imag, 2 * z.real * z.imag)  # z²
                z4 = complex(z2.real * z2.real - z2.imag * z2.imag, 2 * z2.real * z2.imag)  # z⁴
                z = z * z4 + c  # z⁵ + c
            else:
                # For other powers, use cmath.pow with careful handling
                try:
                    z = cmath.pow(z, power) + c
                except (OverflowError, ValueError):
                    # Handle overflow by treating as escaped
                    return 0.0
            
            # Check escape condition with squared magnitude for efficiency
            mag_squared = z.real * z.real + z.imag * z.imag
            
            if mag_squared > escape_radius_squared:
                # Enhanced smooth coloring formula for better visual results
                # Adjust log factor based on power for more consistent gradients
                log_factor = math.log(power) if power > 2 else math.log(2)
                smooth_i = i + 1 - math.log(math.log(mag_squared)) / log_factor
                return smooth_i / max_iter
            
            # Improved numerical instability check
            if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                return 0.0
                
            # For higher powers, add tighter bounds checking to catch divergence earlier
            if power > 3 and mag_squared > 1e10:
                # Value growing too rapidly, treat as escaped
                return 0.0
            
        # If we reached max iterations, point is in the set
        return 1.0
        
    def _julia_numpy(self, x, y, max_iter, c, power):
        """NumPy-accelerated implementation of Julia set calculation"""
        # Starting z is the input coordinate
        z = complex(x, y)
        
        # Get appropriate escape radius for this power
        escape_radius = 2.0 * (1.0 + 0.2 * (power - 2)) if power > 2 else 2.0
        
        for i in range(max_iter):
            # NumPy can handle complex exponentiation efficiently
            if power == 2:
                z = z*z + c  # Most common case
            else:
                z = np.power(z, power) + c
                
            # Check for escape
            if abs(z) > escape_radius:
                # Use numpy's optimized functions for smooth coloring
                mag_squared = np.abs(z)**2
                log_factor = np.log(power) if power > 2 else np.log(2.0)
                smooth_i = i + 1 - np.log(np.log(mag_squared)) / log_factor
                return float(smooth_i / max_iter)
                
            # Check for numerical instability
            if np.isnan(z.real) or np.isnan(z.imag) or np.isinf(z.real) or np.isinf(z.imag):
                return 0.0
                
        # In the set
        return 1.0
    
    @cached_fractal_calc
    def quintic_mandelbulb_value(self, x, y, z, max_iter):
        """Calculate 3D Quintic Mandelbulb value with enhanced stability and optimizations"""
        try:
            # Limit inputs for safety and stability
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            z = safe_value(z, 0, -3, 3)
            
            # Use the stable iteration function for better numerical stability if enabled
            if ENABLE_PATTERN_COHERENCE:
                return self._quintic_mandelbulb_stable(x, y, z, max_iter)
            # Otherwise use optimized implementation based on available libraries
            elif NUMPY_AVAILABLE:
                return self._quintic_mandelbulb_numpy(x, y, z, max_iter)
            else:
                return self._quintic_mandelbulb_standard(x, y, z, max_iter)
                
        except Exception as e:
            debug_print(f"Quintic mandelbulb calculation error: {e}")
            return 0.3
            
    def _quintic_mandelbulb_stable(self, x, y, z, max_iter):
        """More stable implementation of Quintic Mandelbulb using controlled iterations"""
        # Original point
        cx, cy, cz = x, y, z
        # Start point
        px, py, pz = 0, 0, 0
        
        # Quintic power
        power = 5
        
        # Limit iterations and use bailout
        iterations = min(max_iter, 100)
        bailout = 4.0
        
        for i in range(iterations):
            # Check escape condition
            r2 = px*px + py*py + pz*pz
            if r2 > bailout:
                # Enhanced smooth coloring for better gradient
                smooth_i = i + 1 - math.log(math.log(r2)) / math.log(power)
                return smooth_i / iterations
                
            # Check for numerical instability
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
                
            # Use stable iteration function to calculate next point
            nx, ny, nz = stable_mandelbulb_iteration(px, py, pz, power)
            
            # Add constant (equivalent to c in standard formulas)
            px = nx + cx
            py = ny + cy
            pz = nz + cz
            
            # Additional safety check for extreme values
            max_coord = max(abs(px), abs(py), abs(pz))
            if max_coord > 1e10:
                return 0.0
        
        # Inside the set - higher than zero for better visualization
        return 0.05
    
    def _quintic_mandelbulb_standard(self, x, y, z, max_iter):
        """Standard Python implementation of Quintic Mandelbulb with enhanced stability"""
        # Initialize point
        cx, cy, cz = x, y, z  # Original point (c)
        px, py, pz = 0, 0, 0  # Start at origin (p)
        
        # Main iteration loop
        power = 5  # Quintic power
        bailout = 4.0
        
        # Limit iterations for safety
        iterations = min(max_iter, 100)
        
        # Pre-calculate constants for performance
        power_minus_1 = power - 1
        
        for i in range(iterations):
            # Calculate squared radius more efficiently
            r2 = px*px + py*py + pz*pz
            
            # Early bailout check with smooth coloring transition
            if r2 > bailout:
                # Enhanced smooth coloring for 3D
                smooth_i = i + 1 - math.log(math.log(r2)) / math.log(power)
                return smooth_i / iterations
            
            # Check for NaN or inf
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
            
            # Avoid division by zero with improved threshold
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                # More stable spherical coordinate conversion
                r = math.sqrt(r2)
                
                # More numerically stable theta calculation
                if pz == 0:
                    theta = math.pi / 2  # 90 degrees
                else:
                    xy_dist = math.sqrt(px*px + py*py)
                    theta = math.atan2(xy_dist, pz)
                
                # More numerically stable phi calculation
                if px == 0:
                    phi = math.pi / 2 if py > 0 else -math.pi / 2
                else:
                    phi = math.atan2(py, px)
            
            # Calculate r^power with safeguards against extreme values
            # Use math.pow for better precision and safety with quintic power
            r_pow = min(math.pow(r, power), 1000.0)
            
            # Calculate new point in spherical coords
            theta_new = theta * power
            phi_new = phi * power
            
            # Use safer trig functions with bounds checking
            sin_theta = math.sin(theta_new)
            if math.isnan(sin_theta) or math.isinf(sin_theta):
                sin_theta = 0
                
            cos_theta = math.cos(theta_new)
            if math.isnan(cos_theta) or math.isinf(cos_theta):
                cos_theta = 1
                
            sin_phi = math.sin(phi_new)
            if math.isnan(sin_phi) or math.isinf(sin_phi):
                sin_phi = 0
                
            cos_phi = math.cos(phi_new)
            if math.isnan(cos_phi) or math.isinf(cos_phi):
                cos_phi = 1
            
            # Convert back to cartesian coords with controlled magnitude
            px_new = r_pow * sin_theta * cos_phi + cx
            py_new = r_pow * sin_theta * sin_phi + cy
            pz_new = r_pow * cos_theta + cz
            
            # Check for extreme growth or NaN/inf and apply limiting
            if (math.isnan(px_new) or math.isnan(py_new) or math.isnan(pz_new) or
                math.isinf(px_new) or math.isinf(py_new) or math.isinf(pz_new)):
                return 0.0
                
            # Apply limiting for numerical stability while preserving direction
            max_coord = max(abs(px_new), abs(py_new), abs(pz_new))
            if max_coord > 1e10:
                scale_factor = 1e10 / max_coord
                px = px_new * scale_factor
                py = py_new * scale_factor
                pz = pz_new * scale_factor
            else:
                px, py, pz = px_new, py_new, pz_new
        
        # Inside the set - with slight offset from zero for better visualization
        return 0.05
        
    def _quintic_mandelbulb_numpy(self, x, y, z, max_iter):
        """NumPy optimized implementation of Quintic Mandelbulb"""
        # Initialize point
        c = np.array([x, y, z], dtype=np.float64)  # Original point as np array
        p = np.zeros(3, dtype=np.float64)  # Start at origin
        
        # Main iteration loop
        power = 5.0  # Quintic power
        bailout = 4.0
        
        # Limit iterations for safety
        iterations = min(max_iter, 100)
        
        for i in range(iterations):
            # Calculate squared radius
            r2 = np.sum(p**2)
            
            # Early bailout check
            if r2 > bailout:
                # NumPy optimized smooth coloring
                smooth_i = i + 1 - np.log(np.log(r2)) / np.log(power)
                return float(smooth_i / iterations)
            
            # Check for numerical instability
            if np.isnan(r2) or np.isinf(r2):
                return 0.0
            
            # Avoid division by zero with improved threshold
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                # More stable spherical coordinate conversion with NumPy
                r = np.sqrt(r2)
                
                # More numerically stable theta calculation
                xy_dist = np.sqrt(p[0]**2 + p[1]**2)
                theta = np.arctan2(xy_dist, p[2])
                phi = np.arctan2(p[1], p[0])
            
            # Calculate new point in spherical coords
            r_pow = min(r**power, 1000.0)
            theta_new = theta * power
            phi_new = phi * power
            
            # Convert back to cartesian with NumPy functions
            sin_theta = np.sin(theta_new)
            cos_theta = np.cos(theta_new)
            sin_phi = np.sin(phi_new)
            cos_phi = np.cos(phi_new)
            
            # Create new vector with controlled magnitude
            p_new = np.array([
                r_pow * sin_theta * cos_phi + c[0],
                r_pow * sin_theta * sin_phi + c[1],
                r_pow * cos_theta + c[2]
            ])
            
            # Check for numerical issues
            if np.any(np.isnan(p_new)) or np.any(np.isinf(p_new)):
                return 0.0
                
            # Apply limiting for numerical stability
            max_coord = np.max(np.abs(p_new))
            if max_coord > 1e10:
                p = p_new * (1e10 / max_coord)
            else:
                p = p_new
        
        # Inside the set
        return 0.05
    
    @cached_fractal_calc
    def cubic_mandelbulb_value(self, x, y, z, max_iter):
        """Calculate 3D Cubic Mandelbulb value with enhanced stability and optimizations"""
        try:
            # Limit inputs for safety and stability
            x = safe_value(x, 0, -3, 3)
            y = safe_value(y, 0, -3, 3)
            z = safe_value(z, 0, -3, 3)
            
            # Use the stable iteration function for better numerical stability if enabled
            if ENABLE_PATTERN_COHERENCE:
                return self._cubic_mandelbulb_stable(x, y, z, max_iter)
            # Otherwise, use optimized implementation if available
            elif NUMPY_AVAILABLE:
                return self._cubic_mandelbulb_numpy(x, y, z, max_iter)
            else:
                return self._cubic_mandelbulb_standard(x, y, z, max_iter)
                
        except Exception as e:
            debug_print(f"Cubic mandelbulb calculation error: {e}")
            return 0.3
            
    def _cubic_mandelbulb_stable(self, x, y, z, max_iter):
        """More stable implementation of Cubic Mandelbulb using controlled iterations"""
        # Original point
        cx, cy, cz = x, y, z
        # Start point
        px, py, pz = 0, 0, 0
        
        # Cubic power
        power = 3
        
        # Limit iterations and use bailout
        iterations = min(max_iter, 100)
        bailout = 4.0
        
        for i in range(iterations):
            # Check escape condition
            r2 = px*px + py*py + pz*pz
            if r2 > bailout:
                # Enhanced smooth coloring for better gradient
                smooth_i = i + 1 - math.log(math.log(r2)) / math.log(power)
                return smooth_i / iterations
                
            # Check for numerical instability
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
                
            # Use stable iteration function to calculate next point
            nx, ny, nz = stable_mandelbulb_iteration(px, py, pz, power)
            
            # Add constant (equivalent to c in standard formulas)
            px = nx + cx
            py = ny + cy
            pz = nz + cz
            
            # Additional safety check for extreme values
            max_coord = max(abs(px), abs(py), abs(pz))
            if max_coord > 1e10:
                return 0.0
        
        # Inside the set - higher than zero for better visualization
        return 0.05
            
    def _cubic_mandelbulb_standard(self, x, y, z, max_iter):
        """Standard Python implementation of Cubic Mandelbulb with enhanced stability"""
        # Initialize point
        cx, cy, cz = x, y, z  # Original point (c)
        px, py, pz = 0, 0, 0  # Start at origin (p)
        
        # Main iteration loop
        power = 3  # Cubic power - more stable than quintic
        bailout = 4.0
        
        # Limit iterations for safety
        iterations = min(max_iter, 100)
        
        for i in range(iterations):
            # Calculate squared radius with optimized pow
            r2 = px*px + py*py + pz*pz
            
            # Early bailout check with smooth coloring
            if r2 > bailout:
                # Enhanced smooth coloring for better visualization
                smooth_i = i + 1 - math.log(math.log(r2)) / math.log(power)
                return smooth_i / iterations
            
            # Check for NaN or inf
            if math.isnan(r2) or math.isinf(r2):
                return 0.0
            
            # Avoid division by zero with improved handling
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                # More stable spherical coordinate conversion
                r = math.sqrt(r2)
                
                # More numerically stable theta calculation
                if pz == 0:
                    theta = math.pi / 2  # 90 degrees
                else:
                    xy_dist = math.sqrt(px*px + py*py)
                    theta = math.atan2(xy_dist, pz)
                
                # More numerically stable phi calculation
                if px == 0:
                    phi = math.pi / 2 if py > 0 else -math.pi / 2
                else:
                    phi = math.atan2(py, px)
            
            # Calculate r^power with cubic optimization specific to power=3
            # More stable than general pow() for this specific case
            r_pow = r * r * r
            r_pow = min(r_pow, 1000.0)  # Safety cap
            
            # Calculate new point in spherical coords
            theta_new = theta * power
            phi_new = phi * power
            
            # Safer trig functions with bounds checking
            sin_theta = math.sin(theta_new)
            cos_theta = math.cos(theta_new)
            sin_phi = math.sin(phi_new)
            cos_phi = math.cos(phi_new)
            
            # Convert back to cartesian coords
            px_new = r_pow * sin_theta * cos_phi + cx
            py_new = r_pow * sin_theta * sin_phi + cy
            pz_new = r_pow * cos_theta + cz
            
            # Check for extreme values and apply limiting
            if (math.isnan(px_new) or math.isnan(py_new) or math.isnan(pz_new) or
                math.isinf(px_new) or math.isinf(py_new) or math.isinf(pz_new)):
                return 0.0
                
            # Apply limiting for numerical stability while preserving direction
            max_coord = max(abs(px_new), abs(py_new), abs(pz_new))
            if max_coord > 1e10:
                scale_factor = 1e10 / max_coord
                px = px_new * scale_factor
                py = py_new * scale_factor
                pz = pz_new * scale_factor
            else:
                px, py, pz = px_new, py_new, pz_new
        
        # Inside the set - with slight offset from zero for better visualization
        return 0.05
        
    def _cubic_mandelbulb_numpy(self, x, y, z, max_iter):
        """NumPy optimized implementation of Cubic Mandelbulb"""
        # Initialize point
        c = np.array([x, y, z], dtype=np.float64)  # Original point as np array
        p = np.zeros(3, dtype=np.float64)  # Start at origin
        
        # Main iteration loop
        power = 3.0  # Cubic power
        bailout = 4.0
        
        # Limit iterations for safety
        iterations = min(max_iter, 100)
        
        for i in range(iterations):
            # Calculate squared radius
            r2 = np.sum(p**2)
            
            # Early bailout check
            if r2 > bailout:
                # NumPy optimized smooth coloring
                smooth_i = i + 1 - np.log(np.log(r2)) / np.log(power)
                return float(smooth_i / iterations)
            
            # Check for numerical instability
            if np.isnan(r2) or np.isinf(r2):
                return 0.0
            
            # Avoid division by zero
            if r2 < 0.000001:
                r = 0.000001
                theta = 0
                phi = 0
            else:
                # More stable spherical coordinate conversion
                r = np.sqrt(r2)
                xy_dist = np.sqrt(p[0]**2 + p[1]**2)
                theta = np.arctan2(xy_dist, p[2])
                phi = np.arctan2(p[1], p[0])
            
            # Calculate new point in spherical coords
            r_pow = min(r**power, 1000.0)
            theta_new = theta * power
            phi_new = phi * power
            
            # Convert back to cartesian with NumPy functions
            sin_theta = np.sin(theta_new)
            cos_theta = np.cos(theta_new)
            sin_phi = np.sin(phi_new)
            cos_phi = np.cos(phi_new)
            
            # Create new vector with controlled magnitude
            p_new = np.array([
                r_pow * sin_theta * cos_phi + c[0],
                r_pow * sin_theta * sin_phi + c[1],
                r_pow * cos_theta + c[2]
            ])
            
            # Check for numerical issues
            if np.any(np.isnan(p_new)) or np.any(np.isinf(p_new)):
                return 0.0
                
            # Apply limiting for numerical stability
            max_coord = np.max(np.abs(p_new))
            if max_coord > 1e10:
                p = p_new * (1e10 / max_coord)
            else:
                p = p_new
        
        # Inside the set
        return 0.05

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
        """Process a batch of faces with fractal operations - with enhanced performance monitoring"""
        try:
            # Time tracking for batch processing performance
            batch_start_time = time.time()
            
            # Get the batch
            batch = self.face_batches[batch_idx]
            scene = context.scene
            
            # Make sure we're in edit mode
            if self.obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
                self.bm = bmesh.from_edit_mesh(self.mesh)
                ensure_bmesh_lookup_tables(self.bm)
                
            # Apply face coherence selection if enabled
            if ENABLE_PATTERN_COHERENCE:
                batch = select_coherent_faces(self.bm, batch, max_faces=len(batch))
                
            # Special batch grouping for optimization
            # Group faces by proximity in 3D space for better cache locality
            grouped_faces = self._group_faces_by_proximity(batch)
            
            # Process each proximity group in sequence
            for face_group in grouped_faces:
                # Check time elapsed in this batch - if exceeding max batch time,
                # stop processing this batch to keep UI responsive
                time_in_batch = time.time() - batch_start_time
                if time_in_batch > MAX_BATCH_PROCESSING_TIME:
                    debug_print(f"Batch taking too long ({time_in_batch:.2f}s), deferring remaining faces to next batch")
                    # Move remaining faces to a new batch at the end
                    remaining_faces = []
                    for remaining_group in face_group:
                        remaining_faces.extend(remaining_group)
                    if remaining_faces:
                        self.face_batches.append(remaining_faces)
                    break
                    
                # Process each face in the current proximity group 
                # Collect faces and calculate values with coherence
                face_list = []
                face_values = []
                face_centers = []
                
                # First pass - collect all valid faces and calculate initial fractal values
                for face in face_group:
                    # Skip if face is no longer valid
                    if not validate_face(face):
                        continue
                    
                    # Calculate face center
                    center = face.calc_center_median()
                    face_centers.append(center)
                    
                    # Enhanced symmetry optimization
                    if scene.use_symmetry:
                        # Get symmetry properties based on fractal type and power
                        if scene.fractal_type.startswith('JULIA'):
                            # For Julia sets, get symmetry based on power parameter
                            if scene.fractal_type == 'JULIA':
                                power = scene.fractal_power
                            elif scene.fractal_type == 'JULIA_CUBIC':
                                power = 3
                            elif scene.fractal_type == 'JULIA_QUARTIC':
                                power = 4
                            elif scene.fractal_type == 'JULIA_QUINTIC':
                                power = 5
                            else:
                                power = 2
                                
                            symmetry = self.apply_symmetry(scene.fractal_type, power)
                        else:
                            # For other fractal types
                            symmetry = self.apply_symmetry(scene.fractal_type)
                            
                        # Advanced symmetry optimizations based on fractal type
                        fractal_value = None
                        
                        if symmetry['symmetry_type'] == 'MIRROR_X' and center.y < 0:
                            # For Mandelbrot with y-axis mirror symmetry, reflect point across x-axis
                            mirror_center = Vector((center.x, -center.y, center.z))
                            fractal_value = self.get_fractal_value(
                                mirror_center, scene.fractal_type, scene
                            )
                            
                        elif symmetry['symmetry_type'] == 'ROTATIONAL' and symmetry['fold'] > 1:
                            # For rotational symmetry, use distance from origin and angle
                            # to reduce calculations
                            dist = math.sqrt(center.x*center.x + center.y*center.y)
                            if dist > 0:
                                # Get normalized angle in [0, 2π/n] where n is fold count
                                angle = math.atan2(center.y, center.x)
                                fold = symmetry['fold']
                                sector_angle = 2 * math.pi / fold
                                
                                # Normalize angle to first sector
                                norm_angle = angle % sector_angle
                                
                                # Calculate point in first sector with same distance from origin
                                norm_x = dist * math.cos(norm_angle)
                                norm_y = dist * math.sin(norm_angle)
                                
                                # Get fractal value for normalized point
                                norm_center = Vector((norm_x, norm_y, center.z))
                                fractal_value = self.get_fractal_value(
                                    norm_center, scene.fractal_type, scene
                                )
                            
                        # If no symmetry-based value calculated, calculate normally
                        if fractal_value is None:
                            fractal_value = self.get_fractal_value(
                                center, scene.fractal_type, scene
                            )
                    else:
                        # No symmetry optimization, calculate normally
                        fractal_value = self.get_fractal_value(
                            center, scene.fractal_type, scene
                        )
                    
                    # Add to our lists
                    face_list.append(face)
                    face_values.append(fractal_value)
                
                # Apply pattern coherence if enabled
                if ENABLE_PATTERN_COHERENCE and len(face_list) > 1:
                    face_values = apply_coherence_pattern(face_list, face_values, scene.fractal_type)
                    
                # Second pass - apply extrusions using coherent values
                for i, (face, fractal_value) in enumerate(zip(face_list, face_values)):
                    # Apply extrusion based on fractal value with enhanced thresholding
                    # Dynamic threshold based on fractal type - higher for complex 3D fractals
                    threshold = 0.1  # Default
                    if scene.fractal_type.endswith('MANDELBULB'):
                        threshold = 0.15  # Higher threshold for 3D fractals for stability
                        
                    if fractal_value > threshold:
                        try:
                            # Find neighboring faces for normal alignment
                            neighboring_faces = []
                            if ENABLE_PATTERN_COHERENCE:
                                # Get adjacent faces
                                for edge in face.edges:
                                    for adjacent_face in edge.link_faces:
                                        if adjacent_face != face and adjacent_face in face_list:
                                            neighboring_faces.append(adjacent_face)
                                            # Limit number of neighbors to consider
                                            if len(neighboring_faces) >= MAX_COHERENCE_NEIGHBORS:
                                                break
                                    if len(neighboring_faces) >= MAX_COHERENCE_NEIGHBORS:
                                        break
                            
                            # Use extrude → insert → extrude pattern with progress updates
                            success = self.process_complex_pattern(face, fractal_value, scene, neighboring_faces)
                            if success:
                                self.extruded_faces += 1
                                
                                # Periodically update progress during batch processing
                                if self.extruded_faces % 20 == 0 and hasattr(context.window_manager, 'fractal_progress'):
                                    # Calculate overall progress more accurately
                                    total_processed = self.current_batch * len(batch) + self.processed_faces
                                    total_faces = sum(len(b) for b in self.face_batches)
                                    progress = (total_processed / total_faces) * 100 if total_faces > 0 else 0
                                    context.window_manager.fractal_progress = progress
                                    
                        except Exception as e:
                            debug_print(f"Error extruding face: {e}")
                    
                    self.processed_faces += 1
                    
                    # Enhanced safety check with progressive mesh decimation if needed
                    verts_count = len(self.bm.verts)
                    if verts_count > MAX_SAFE_VERTICES * 0.9:  # Approaching limit
                        if verts_count > MAX_SAFE_VERTICES:
                            self.report({'WARNING'}, f"Vertex limit reached ({verts_count}), stopping batch")
                            # Force mesh update before breaking
                            bmesh.update_edit_mesh(self.mesh)
                            break
                        
                        # When getting close to vertex limit, start decimating mesh
                        # to maintain stability while continuing processing
                        if verts_count > MAX_SAFE_VERTICES * 0.95:
                            debug_print("Approaching vertex limit, performing decimation")
                            try:
                                # Perform light decimation to reduce vertex count
                                # Collapse short edges to reduce complexity without changing appearance much
                                bmesh.ops.dissolve_degenerate(
                                    self.bm,
                                    dist=0.0001,
                                    edges=self.bm.edges
                                )
                                bmesh.update_edit_mesh(self.mesh)
                            except Exception as decimate_error:
                                debug_print(f"Decimation error: {decimate_error}")
            
            # Calculate batch processing time for performance monitoring
            batch_time = time.time() - batch_start_time
            if batch_time > 1.0:  # Only log slow batches
                debug_print(f"Batch {batch_idx+1}/{len(self.face_batches)} took {batch_time:.2f}s")
                
            # Only update mesh after all faces in batch are processed
            # This reduces the number of expensive mesh updates
            bmesh.update_edit_mesh(self.mesh)
            
            # Force redraw but limit frequency to improve performance
            if context.area and (batch_idx % 2 == 0 or batch_idx == len(self.face_batches) - 1):
                context.area.tag_redraw()
                
            return True
            
        except Exception as e:
            debug_print(f"Error processing batch: {e}")
            if DEBUG:
                traceback.print_exc()
            return False
    
    def _group_faces_by_proximity(self, faces, max_groups=4):
        """Group faces by proximity in 3D space for better cache locality and processing efficiency"""
        if not faces:
            return []
            
        try:
            # For very small batches, just return the whole batch as one group
            if len(faces) < 20:
                return [faces]
                
            # Calculate the center of each face
            face_centers = []
            for face in faces:
                if validate_face(face):
                    center = face.calc_center_median()
                    face_centers.append((face, center))
                    
            # Simple grouping by octants (dividing 3D space into 8 regions)
            # This is faster than clustering algorithms for our purpose
            octants = [[] for _ in range(8)]
            
            # Determine the center of all faces
            if face_centers:
                avg_x = sum(c.x for _, c in face_centers) / len(face_centers)
                avg_y = sum(c.y for _, c in face_centers) / len(face_centers)
                avg_z = sum(c.z for _, c in face_centers) / len(face_centers)
                
                # Group faces by octant relative to the average center
                for face, center in face_centers:
                    # Determine octant (0-7) based on position relative to average
                    octant_idx = 0
                    if center.x >= avg_x: octant_idx |= 1
                    if center.y >= avg_y: octant_idx |= 2
                    if center.z >= avg_z: octant_idx |= 4
                    
                    octants[octant_idx].append(face)
            
            # Filter out empty groups and ensure we don't have too many groups
            groups = [group for group in octants if group]
            
            # If we have too many small groups, consolidate them
            if len(groups) > max_groups:
                # Sort groups by size and keep the largest ones
                groups.sort(key=len, reverse=True)
                groups = groups[:max_groups]
                
            # If no valid groups were created, return original faces as one group
            if not groups:
                return [faces]
                
            return groups
            
        except Exception as e:
            debug_print(f"Error grouping faces: {e}")
            # Fall back to single group if grouping fails
            return [faces]
    
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
            
            # Update global coherence settings from scene properties
            global ENABLE_VALUE_NORMALIZATION, ENABLE_PATTERN_COHERENCE, ENABLE_EXTRUSION_CONTROL, NEIGHBOR_INFLUENCE
            ENABLE_VALUE_NORMALIZATION = scene.fractal_enable_value_normalization
            ENABLE_PATTERN_COHERENCE = scene.fractal_enable_coherence
            ENABLE_EXTRUSION_CONTROL = scene.fractal_enable_extrusion_control
            NEIGHBOR_INFLUENCE = scene.fractal_coherence_strength
            
            # Clear cache on execution for fresh start
            global FRACTAL_CACHE
            FRACTAL_CACHE.clear()
            
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
            
            # Apply coherent face selection if enabled
            if ENABLE_PATTERN_COHERENCE:
                faces_to_process = select_coherent_faces(self.bm, faces_to_process, max_faces=face_limit)
            
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
    
    # Visual coherence properties
    bpy.types.Scene.fractal_enable_value_normalization = BoolProperty(
        name="Value Normalization",
        description="Apply non-linear normalization to fractal values for more predictable results",
        default=True,
        update=lambda self, context: setattr(context.scene, "force_coherence_update", True)
    )
    bpy.types.Scene.fractal_enable_coherence = BoolProperty(
        name="Enable Pattern Coherence",
        description="Create smoother transitions between neighboring faces",
        default=True,
        update=lambda self, context: setattr(context.scene, "force_coherence_update", True)
    )
    bpy.types.Scene.fractal_coherence_strength = FloatProperty(
        name="Coherence Strength",
        description="How strongly neighboring faces influence each other",
        default=0.3,
        min=0.0,
        max=1.0,
        precision=2,
        subtype='FACTOR',
        update=lambda self, context: setattr(context.scene, "force_coherence_update", True)
    )
    bpy.types.Scene.fractal_enable_extrusion_control = BoolProperty(
        name="Controlled Extrusion",
        description="Use more consistent extrusion parameterization for better results",
        default=True,
        update=lambda self, context: setattr(context.scene, "force_coherence_update", True)
    )
    bpy.types.Scene.force_coherence_update = BoolProperty(
        default=False
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
    
    # Remove coherence properties
    del bpy.types.Scene.fractal_enable_value_normalization
    del bpy.types.Scene.fractal_enable_coherence
    del bpy.types.Scene.fractal_coherence_strength
    del bpy.types.Scene.fractal_enable_extrusion_control
    del bpy.types.Scene.force_coherence_update

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
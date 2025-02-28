bl_info = {
    "name": "Fractal Geometry Generator",
    "author": "Your Name",
    "version": (0, 4, 0),
    "blender": (4, 3, 0),  # Updated for Blender 4.3.x compatibility
    "location": "View3D > Sidebar > Fractal",
    "description": "Generate fractal-based geometry modifications",
    "warning": "Experimental - Use with caution",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import bmesh
import random
import math
import time
from mathutils import Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    FloatVectorProperty
)
import traceback  # For better error reporting

# Fractal types for more pattern options
FRACTAL_TYPES = [
    ('MANDELBROT', "Mandelbrot", "Classic Mandelbrot set pattern"),
    ('JULIA', "Julia", "Julia set with configurable parameters"),
    ('PLASMA', "Plasma", "Plasma fractal with smooth transitions")
]

# Extrusion direction options
EXTRUSION_DIRECTIONS = [
    ('NORMAL', "Face Normal", "Extrude along face normal"),
    ('ORIGINAL', "Original Normal", "Extrude along original face normal"),
    ('RANDOM', "Randomized", "Extrude along slightly randomized normal")
]

# Debug flag - set to True for verbose logging
DEBUG = True

def debug_print(message):
    """Print debug messages if debug is enabled"""
    if DEBUG:
        print(f"FRACTAL DEBUG: {message}")

def validate_face(face):
    """Comprehensive validation for a BMesh face for Blender 4.x"""
    if not face or not hasattr(face, "is_valid"):
        return False
    
    try:
        if not face.is_valid:
            return False
        
        # Check if the face has at least 3 valid vertices
        if len(face.verts) < 3:
            return False
            
        # Check if all vertices and edges are valid
        for vert in face.verts:
            if not vert.is_valid:
                return False
        
        for edge in face.edges:
            if not edge.is_valid:
                return False
        
        # Check face area
        try:
            face_area = face.calc_area()
            if face_area < 0.0001:  # Skip very tiny faces
                return False
        except:
            return False
        
        # Check if normal is valid
        try:
            normal = face.normal
            if normal.length < 0.0001:
                return False
        except:
            return False
        
        return True
    except Exception as e:
        if DEBUG:
            debug_print(f"Face validation error: {e}")
        return False

def ensure_bmesh_lookup_tables(bm):
    """Ensure all BMesh lookup tables are valid"""
    try:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        return True
    except Exception as e:
        debug_print(f"Error updating lookup tables: {e}")
        return False

def safe_mode_set(mode, obj=None):
    """Safely set the object mode with error handling"""
    try:
        # In Blender 4.x, the mode_set behavior can be slightly different
        # Always check current mode to avoid redundant mode changes
        if not obj:
            obj = bpy.context.active_object
            
        if obj and obj.mode != mode:
            # Store selection state
            was_selected = obj.select_get()
            
            # Ensure object is active and selected
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Set mode
            bpy.ops.object.mode_set(mode=mode)
            
            # Restore selection if needed
            if not was_selected:
                obj.select_set(False)
                
            return True
    except Exception as e:
        debug_print(f"Mode set error: {e}")
        traceback.print_exc()
        return False
    return True

def force_update_mesh(obj):
    """Force update the mesh data in Blender 4.3.2"""
    try:
        # This is a trick to force Blender to update the mesh
        if obj.mode == 'EDIT':
            # Toggle to object mode and back to force update
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='EDIT')
        return True
    except Exception as e:
        debug_print(f"Mesh update error: {e}")
        traceback.print_exc()
        return False

# CRITICALLY IMPORTANT FUNCTION FOR SELECTION
def get_selected_faces_431(obj):
    """Special function for Blender 4.3.1+ to get selected faces"""
    selected_faces = []
    debug_print(f"Using special 4.3.1+ selection detection")
    
    try:
        # First ensure we're in edit mode
        if obj.mode != 'EDIT':
            safe_mode_set('EDIT', obj)
            
        # Get the mesh
        mesh = obj.data
        
        # Get BMesh (this captures current selection state)
        bm_new = bmesh.from_edit_mesh(mesh)
        ensure_bmesh_lookup_tables(bm_new)
        
        # First approach: Direct selection flag check on faces
        selected_faces = [f for f in bm_new.faces if f.select and validate_face(f)]
        debug_print(f"Direct selection check found {len(selected_faces)} faces")
        
        # If nothing found, try alternative approaches
        if not selected_faces:
            # Toggle selection to force update
            debug_print("Trying selection toggle trick")
            bpy.ops.mesh.select_all(action='INVERT')
            bpy.ops.mesh.select_all(action='INVERT')
            
            # Update BMesh from mesh
            bmesh.update_edit_mesh(mesh)
            
            # Try again with refreshed selection state
            selected_faces = [f for f in bm_new.faces if f.select and validate_face(f)]
            debug_print(f"After toggle trick found {len(selected_faces)} faces")
            
        # Final approach: If available, use selection history
        if not selected_faces and hasattr(bm_new, "select_history") and bm_new.select_history:
            debug_print("Trying selection history")
            for elem in bm_new.select_history:
                if isinstance(elem, bmesh.types.BMFace) and validate_face(elem):
                    selected_faces.append(elem)
        
        # Log final result
        debug_print(f"Final selection count: {len(selected_faces)}")
        return selected_faces, bm_new
        
    except Exception as e:
        debug_print(f"Selection detection error: {e}")
        traceback.print_exc()
        return [], None

class FractalGeometryProcessor:
    """Handles all fractal geometry operations"""
    
    def __init__(self, context, bm):
        debug_print("Initializing FractalGeometryProcessor")
        self.context = context
        self.bm = bm
        self.scene = context.scene
        self.timer = time.time()
        self.original_normals = {}  # Store original face normals
        self.processed_count = 0
        self.failed_count = 0
        
    def log_progress(self, message):
        """Debug logging with timing"""
        current = time.time()
        elapsed = current - self.timer
        self.timer = current
        debug_print(f"{message} ({elapsed:.2f}s)")
        
    def store_original_normal(self, face):
        """Store original face normal for later use"""
        if face.is_valid:
            self.original_normals[face.index] = face.normal.copy()
        
    def get_original_normal(self, face):
        """Get stored original normal for a face"""
        if face.index in self.original_normals:
            return self.original_normals[face.index]
        return face.normal.copy()
        
    def calculate_fractal_value(self, x, y, z=0):
        """Calculate fractal value based on selected type"""
        fractal_type = self.scene.fractal_type
        
        if fractal_type == 'MANDELBROT':
            return self._calculate_mandelbrot(x, y)
        elif fractal_type == 'JULIA':
            return self._calculate_julia(x, y)
        elif fractal_type == 'PLASMA':
            return self._calculate_plasma(x, y, z)
        return 0.5  # Default fallback
    
    def _calculate_mandelbrot(self, x, y):
        """Optimized Mandelbrot calculation with smooth coloring"""
        # Scale around interesting region
        x = x * 2.5 * self.scene.fractal_scale - 0.7
        y = y * 2.5 * self.scene.fractal_scale
        
        c = complex(x, y)
        z = 0
        
        # Early bailout optimization
        for i in range(self.scene.fractal_iterations):
            z = z * z + c
            if abs(z) > 4:
                # Smooth coloring formula
                return (i + 1 - math.log(math.log(abs(z))) / math.log(2)) / self.scene.fractal_iterations
        return 1.0
    
    def _calculate_julia(self, x, y):
        """Julia set calculation with seed parameter"""
        # Get Julia seed from user parameters
        seed_x = self.scene.fractal_julia_seed[0]
        seed_y = self.scene.fractal_julia_seed[1]
        
        x = x * 3.0 * self.scene.fractal_scale
        y = y * 3.0 * self.scene.fractal_scale
        
        z = complex(x, y)
        c = complex(seed_x, seed_y)
        
        for i in range(self.scene.fractal_iterations):
            z = z * z + c
            if abs(z) > 4:
                # Smooth coloring formula
                return (i + 1 - math.log(math.log(abs(z))) / math.log(2)) / self.scene.fractal_iterations
        return 1.0
    
    def _calculate_plasma(self, x, y, z):
        """Plasma fractal for more organic patterns"""
        scale = self.scene.fractal_scale * 5.0
        value = 0
        amplitude = 1.0
        
        for i in range(min(8, self.scene.fractal_iterations // 6)):
            frequency = 2 ** i
            value += self._perlin_noise(x * scale * frequency, 
                                       y * scale * frequency,
                                       z * scale * frequency) * amplitude
            amplitude *= 0.5
            
        return (math.sin(value * 3.14159) + 1) * 0.5
    
    def _perlin_noise(self, x, y, z):
        """Simple Perlin-like noise function"""
        # Simple deterministic noise function for demonstration
        # A real implementation would use a proper noise library
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)
        
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)
        
        # Simplified noise calculation
        hash_val = (X + Y * 37 + Z * 157) % 255
        return ((hash_val / 255.0) - 0.5) * 2.0
    
    def _fade(self, t):
        """Smoothing function for Perlin noise"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def process_face(self, face):
        """Process a single face with fractal operations"""
        # Use the validation function
        if not validate_face(face):
            debug_print(f"Skipping invalid face")
            self.failed_count += 1
            return False
            
        debug_print(f"Processing face with {len(face.verts)} vertices, area: {face.calc_area():.6f}")
        
        # Store the original normal for possible use later
        self.store_original_normal(face)
            
        try:
            # Calculate center point for fractal sampling
            center = face.calc_center_median()
            debug_print(f"Face center: {center.x:.2f}, {center.y:.2f}, {center.z:.2f}")
            
            # Calculate face area for scaling operations
            face_area = face.calc_area()
                
            # Calculate fractal value based on position
            fractal_value = self.calculate_fractal_value(
                center.x, center.y, center.z
            )
            debug_print(f"Fractal value: {fractal_value:.4f}")
            
            # Skip if below minimum depth
            if fractal_value <= self.scene.fractal_min_depth:
                debug_print(f"Skipping face - below minimum depth")
                return False
                
            # Perform primary inset
            debug_print(f"Performing primary inset with depth {self.scene.fractal_inset_depth}")
            primary_faces = self.create_inset(face, fractal_value)
            if not primary_faces:
                debug_print("Primary inset failed - no faces returned")
                self.failed_count += 1
                return False
                
            debug_print(f"Primary inset created {len(primary_faces)} faces")
                
            # Process each created face
            for new_face in primary_faces:
                if not validate_face(new_face):
                    continue
                    
                if random.random() < self.scene.fractal_recursion_chance:
                    debug_print("Performing secondary processing")
                    self.process_secondary_face(new_face, fractal_value)
                    
            self.processed_count += 1
            return True
        
        except Exception as e:
            debug_print(f"Error processing face: {e}")
            if DEBUG:
                traceback.print_exc()
            self.failed_count += 1
            return False

    def create_inset(self, face, fractal_value):
        """Create the primary inset on a face with Blender 4.x compatibility"""
        try:
            # Calculate thickness based on face size for more consistent results
            face_area = face.calc_area()
            thickness = fractal_value * self.scene.fractal_inset_depth
            
            # Scale thickness by square root of area for proportional sizing
            if self.scene.fractal_adapt_to_size:
                thickness *= math.sqrt(face_area) * 0.25  # Reduced from 0.5 to 0.25 for Blender 4.3.2
            
            # For Blender 4.x, we need to ensure thickness is reasonable
            # Cap thickness to prevent creating degenerate geometry
            max_inset = min(face_area * 0.25, 0.5)  # More conservative for Blender 4.3.2
            thickness = min(thickness, max_inset)
            
            # Absolute minimum thickness to avoid precision issues
            thickness = max(thickness, 0.0001)
                
            # Apply the inset operation with Blender 4.x parameters
            result = bmesh.ops.inset_region(
                self.bm,
                faces=[face],
                thickness=thickness,
                depth=0,
                use_boundary=True,
                use_even_offset=True
            )
            
            # Update BMesh lookups - critical in Blender 4.x
            ensure_bmesh_lookup_tables(self.bm)
            
            # Filter valid faces
            valid_faces = [f for f in result.get('faces', []) if validate_face(f)]
            
            if DEBUG and not valid_faces:
                debug_print(f"Inset created no valid faces (thickness: {thickness}, area: {face_area})")
                
            return valid_faces
            
        except Exception as e:
            debug_print(f"Inset error: {e}")
            if DEBUG:
                traceback.print_exc()
            return []
    
    def process_secondary_face(self, face, fractal_value):
        """Apply secondary operations (inset & extrude) to a face"""
        try:
            # Secondary inset
            thickness = fractal_value * self.scene.fractal_secondary_inset
            
            # Scale by face area if enabled
            face_area = face.calc_area()
            if self.scene.fractal_adapt_to_size:
                thickness *= math.sqrt(face_area) * 0.25  # Reduced for Blender 4.3.2
                
            # Cap thickness for Blender 4.x
            max_inset = min(face_area * 0.2, 0.4)  # More conservative for 4.3.2
            thickness = min(thickness, max_inset)
            
            # Absolute minimum thickness
            thickness = max(thickness, 0.0001)
            
            # Create secondary inset with Blender 4.x compatibility
            result = bmesh.ops.inset_region(
                self.bm,
                faces=[face],
                thickness=thickness,
                depth=0,
                use_boundary=True,
                use_even_offset=True
            )
            
            # Update BMesh lookups
            ensure_bmesh_lookup_tables(self.bm)
            
            # Process the resulting faces
            for inset_face in result.get('faces', []):
                if validate_face(inset_face):
                    # Apply extrusion with Blender 4.x compatibility
                    self.extrude_face_safely(inset_face, fractal_value)
                
        except Exception as e:
            debug_print(f"Secondary processing error: {e}")
            if DEBUG:
                traceback.print_exc()
    
    def extrude_face_safely(self, face, fractal_value):
        """Improved extrusion with Blender 4.3.2 compatibility"""
        try:
            # Calculate extrusion direction based on user setting (if available)
            if hasattr(self.scene, "fractal_extrusion_direction"):
                direction_mode = self.scene.fractal_extrusion_direction
                
                if direction_mode == 'ORIGINAL':
                    # Use original face normal
                    direction = self.get_original_normal(face)
                elif direction_mode == 'RANDOM':
                    # Create a slightly randomized normal
                    face_normal = face.normal.copy()
                    # Generate deterministic random offset from position
                    center = face.calc_center_median()
                    rand_seed = hash(f"{center.x:.2f},{center.y:.2f},{center.z:.2f}")
                    random.seed(rand_seed)
                    rand_vec = Vector((
                        random.uniform(-0.2, 0.2),  # Reduced randomness for 4.3.2
                        random.uniform(-0.2, 0.2),
                        random.uniform(-0.2, 0.2)
                    ))
                    direction = (face_normal + rand_vec).normalized()
                    # Reset the random seed
                    if self.scene.fractal_random_seed == 0:
                        random.seed(None)
                    else:
                        random.seed(self.scene.fractal_random_seed)
                else:  # Default to face normal
                    direction = face.normal.copy()
            else:
                # If property not available, use face normal
                direction = face.normal.copy()
                
            # Ensure direction is normalized
            if direction.length < 0.0001:
                direction = Vector((0, 0, 1))  # Fallback direction
            else:
                direction.normalize()
                
            # Calculate reasonable extrusion height for Blender 4.x
            face_area = face.calc_area()
            
            # Base height from fractal value and user parameter
            height = fractal_value * self.scene.fractal_extrusion_strength
            
            # Add controlled randomness - use position for consistent results
            center = face.calc_center_median()
            # Generate deterministic random value from position
            position_hash = (center.x * 64 + center.y * 128 + center.z * 256) 
            random_factor = (math.sin(position_hash) * 0.5 + 0.5) * 0.5 + 0.75
            height *= random_factor
            
            # Scale by face area if enabled
            if self.scene.fractal_adapt_to_size:
                height *= math.sqrt(face_area) * 0.3  # Reduced for 4.3.2
                
            # Limit to reasonable height for Blender 4.x (more conservative)
            height = min(height, face_area * 1.5)  # More conservative for 4.3.2
            
            # Try to use the most reliable extrusion method for Blender 4.3.2
            # Modified approach:
            extruded_ok = False
            
            # Method 1: Duplicate and translate (most reliable in 4.3.2)
            try:
                # Duplicate the face
                ret = bmesh.ops.duplicate(self.bm, geom=[face])
                dupe_faces = [f for f in ret["geom"] if isinstance(f, bmesh.types.BMFace)]
                
                if dupe_faces:
                    # For each duplicated face, move its vertices
                    for dupe_face in dupe_faces:
                        verts = dupe_face.verts
                        bmesh.ops.translate(
                            self.bm,
                            vec=direction * height,
                            verts=list(verts)
                        )
                    extruded_ok = True
                    
                # Update lookup tables
                ensure_bmesh_lookup_tables(self.bm)
                
            except Exception as e:
                debug_print(f"Duplication method failed: {e}")
                extruded_ok = False
            
            # Method 2: If first method failed, try discrete extrusion
            if not extruded_ok:
                try:
                    result = bmesh.ops.extrude_discrete_faces(
                        self.bm,
                        faces=[face]
                    )
                    
                    # Update lookup tables
                    ensure_bmesh_lookup_tables(self.bm)
                    
                    extruded_faces = result.get('faces', [])
                    if extruded_faces:
                        # For each extruded face, translate it
                        for ext_face in extruded_faces:
                            if validate_face(ext_face):
                                verts = list(ext_face.verts)
                                
                                # Apply translation
                                if verts:
                                    bmesh.ops.translate(
                                        self.bm,
                                        vec=direction * height,
                                        verts=verts
                                    )
                                    extruded_ok = True
                                    
                        # Update lookup tables again
                        ensure_bmesh_lookup_tables(self.bm)
                        
                except Exception as e:
                    debug_print(f"Discrete extrusion failed: {e}")
                    extruded_ok = False
                    
            # Method 3: Last resort, try regular extrusion
            if not extruded_ok:
                try:
                    geom = [face]
                    result = bmesh.ops.extrude_face_region(
                        self.bm,
                        geom=geom
                    )
                    
                    # Update lookup tables
                    ensure_bmesh_lookup_tables(self.bm)
                    
                    # Just extract vertices
                    verts = [v for v in result["geom"] 
                            if isinstance(v, bmesh.types.BMVert)]
                            
                    # Apply translation to the extruded vertices
                    if verts:
                        bmesh.ops.translate(
                            self.bm,
                            vec=direction * height,
                            verts=verts
                        )
                        
                    # Update lookup tables
                    ensure_bmesh_lookup_tables(self.bm)
                    
                except Exception as e:
                    debug_print(f"Regular extrusion failed too: {e}")
                
        except Exception as e:
            debug_print(f"Extrusion error: {e}")
            if DEBUG:
                traceback.print_exc()

class FRACTAL_PT_main_panel(bpy.types.Panel):
    bl_label = "Fractal Generator v0.4"
    bl_idname = "FRACTAL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Fractal'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        wm = context.window_manager
        
        # Version info
        box = layout.box()
        box.label(text="Blender 4.3.2 Compatible", icon='CHECKMARK')
        
        # Debug info
        if DEBUG:
            box = layout.box()
            box.label(text="Debug Mode Active", icon='INFO')
            row = box.row()
            row.operator("mesh.fractal_debug", text="Reset Processing State", icon='FILE_REFRESH')
        
        # Processing indicator
        if hasattr(wm, 'fractal_is_processing') and wm.fractal_is_processing:
            box = layout.box()
            box.alert = True
            box.label(text="Processing Fractal...", icon='SORTTIME')
            if hasattr(wm, 'fractal_progress'):
                progress = wm.fractal_progress
                box.label(text=f"Progress: {progress:.1f}%")
        
        # Face Selection Settings
        box = layout.box()
        box.label(text="Face Selection", icon='FACESEL')
        
        # Add help for selection mode
        if scene.fractal_selected_only:
            selection_box = box.box()
            selection_box.label(text="Selection Mode Active!", icon='INFO')
            selection_box.label(text="1. Tab into Edit Mode")
            selection_box.label(text="2. Select faces")
            selection_box.label(text="3. Press Generate")
        
        box.prop(scene, "fractal_selected_only")
        box.prop(scene, "fractal_face_limit")
        box.prop(scene, "fractal_batch_size")
        
        # Pattern Settings
        box = layout.box()
        box.label(text="Pattern Settings", icon='SHADERFX')
        box.prop(scene, "fractal_type")
        
        # Show Julia seed only when Julia is selected
        if scene.fractal_type == 'JULIA':
            box.prop(scene, "fractal_julia_seed")
        
        box.prop(scene, "fractal_random_seed")
        box.prop(scene, "fractal_iterations")
        box.prop(scene, "fractal_scale")
        box.prop(scene, "fractal_min_depth")
        
        # Geometry Settings
        box = layout.box()
        box.label(text="Geometry Settings", icon='MOD_BEVEL')
        box.prop(scene, "fractal_adapt_to_size")
        box.prop(scene, "fractal_inset_depth")
        box.prop(scene, "fractal_secondary_inset")
        box.prop(scene, "fractal_extrusion_strength")
        
        # Add extrusion direction if available
        if hasattr(scene, "fractal_extrusion_direction"):
            box.prop(scene, "fractal_extrusion_direction")
            
        box.prop(scene, "fractal_recursion_chance")
        
        # Generate Button
        row = layout.row(align=True)
        row.scale_y = 2.0
        
        # If processing, show cancel button
        if hasattr(wm, 'fractal_is_processing') and wm.fractal_is_processing:
            row.operator("mesh.fractal_cancel", text="Cancel Generation", icon='X')
        else:
            row.operator("mesh.fractal_generate", text="Generate Fractal", icon='SHADERFX')

class MESH_OT_fractal_debug(bpy.types.Operator):
    """Debug helper to reset processing state for Blender 4.x"""
    bl_idname = "mesh.fractal_debug"
    bl_label = "Reset Fractal Processing State"
    
    def execute(self, context):
        debug_print("Executing reset operation")
        
        # Force reset the processing flags
        context.window_manager.fractal_is_processing = False
        context.window_manager.fractal_should_cancel = False
        if hasattr(context.window_manager, 'fractal_progress'):
            context.window_manager.fractal_progress = 0.0
        
        # Try to ensure we're in a proper mode
        try:
            obj = context.active_object
            if obj and obj.type == 'MESH':
                # First try to end any modal operation
                try:
                    bpy.ops.mesh.fractal_cancel()
                except:
                    pass
                
                # Wait a moment
                time.sleep(0.1)
                
                # Force return to object mode
                if obj.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                    
                # Force UI update
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()
        except Exception as e:
            debug_print(f"Reset mode error: {e}")
            traceback.print_exc()
            
        debug_print("Processing state reset")
        self.report({'INFO'}, "Fractal processing state has been reset")
        return {'FINISHED'}

class MESH_OT_fractal_generate(bpy.types.Operator):
    bl_idname = "mesh.fractal_generate"
    bl_label = "Generate Fractal"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        debug_print("Checking poll conditions...")
        try:
            obj = context.active_object
            
            if obj is None:
                debug_print("Poll failed: No active object")
                return False
                
            if obj.type != 'MESH':
                debug_print("Poll failed: Active object is not a mesh")
                return False
                
            if hasattr(context.window_manager, 'fractal_is_processing') and context.window_manager.fractal_is_processing:
                debug_print("Poll failed: Already processing")
                return False
                
            # In Blender 4.x, checking selection state needs special care
            if context.scene.fractal_selected_only:
                # Check if we're in edit mode
                if obj.mode != 'EDIT':
                    debug_print("Poll failed: Not in edit mode but 'Selected Faces Only' is enabled")
                    return False
                
                # No specific selection check here - we'll do it in execute
                # This allows the button to be clickable and show a proper error message
            
            debug_print("Poll passed!")
            return True
            
        except Exception as e:
            debug_print(f"Poll error: {e}")
            traceback.print_exc()
            return False
    
    def modal(self, context, event):
        try:
            if event.type == 'ESC' or context.window_manager.fractal_should_cancel:
                debug_print("Modal: Cancelled")
                self.cancel(context)
                return {'CANCELLED'}
            
            if event.type == 'TIMER':
                # Process next batch of faces
                if self.current_batch < len(self.face_batches):
                    debug_print(f"Processing batch {self.current_batch+1}/{len(self.face_batches)}")
                    self.process_batch(context, self.current_batch)
                    self.current_batch += 1
                    
                    # Update progress info
                    progress = (self.current_batch / len(self.face_batches)) * 100
                    if hasattr(context.window_manager, 'fractal_progress'):
                        context.window_manager.fractal_progress = progress
                    self.report({'INFO'}, f"Processing: {progress:.1f}% complete")
                    
                    # In Blender 4.x, we need to force the redraw
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                    
                    return {'RUNNING_MODAL'}
                else:
                    debug_print("Modal: Finished all batches")
                    self.finish(context)
                    return {'FINISHED'}
                    
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            debug_print(f"Modal error: {e}")
            if DEBUG:
                traceback.print_exc()
            self.report({'ERROR'}, f"Error in modal processing: {str(e)}")
            self.cleanup(context)
            return {'CANCELLED'}

    def process_batch(self, context, batch_idx):
        """Process a batch of faces with fractal operations"""
        try:
            batch = self.face_batches[batch_idx]
            processor = self.processor
            
            # In Blender 4.x, make sure we're still in edit mode
            if context.active_object.mode != 'EDIT':
                safe_mode_set('EDIT')
            
            # Process each face in batch
            processed_count = 0
            for face in batch:
                if context.window_manager.fractal_should_cancel:
                    return False
                    
                if validate_face(face):
                    if processor.process_face(face):
                        processed_count += 1
            
            # Update mesh and view - very important in Blender 4.x
            try:
                # For Blender 4.x compatibility
                if self.bm and self.mesh:
                    # Ensure lookups are valid
                    ensure_bmesh_lookup_tables(self.bm)
                    
                    # Update edit mesh
                    bmesh.update_edit_mesh(self.mesh)
                    
                    # Tag redraw as needed
                    if context.area:
                        context.area.tag_redraw()
            except Exception as e:
                debug_print(f"Error updating mesh: {e}")
                
            debug_print(f"Batch {batch_idx+1} processed {processed_count}/{len(batch)} faces")
            return True
            
        except Exception as e:
            debug_print(f"Batch processing error: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            self.report({'ERROR'}, f"Batch processing error: {str(e)}")
            return False

    def execute(self, context):
        debug_print("Execute called for Blender 4.3.2")
        try:
            # Initialize state
            obj = context.active_object
            scene = context.scene
            
            if obj is None:
                debug_print("No active object")
                self.report({'ERROR'}, "No active object")
                return {'CANCELLED'}
                
            debug_print(f"Target object: {obj.name}")
            
            # Set random seed
            seed_value = scene.fractal_random_seed
            if seed_value == 0:
                random.seed(None)
                debug_print("Using time-based random seed")
            else:
                random.seed(seed_value)
                debug_print(f"Using fixed random seed: {seed_value}")
            
            # Initialize processing state
            context.window_manager.fractal_is_processing = True
            context.window_manager.fractal_should_cancel = False
            if hasattr(context.window_manager, 'fractal_progress'):
                context.window_manager.fractal_progress = 0.0
            
            # Store original mode
            self.original_mode = obj.mode
            debug_print(f"Original mode: {self.original_mode}")
            
            # Enter edit mode - in Blender 4.x, this needs to be handled carefully
            if obj.mode != 'EDIT':
                debug_print("Entering edit mode")
                if not safe_mode_set('EDIT', obj):
                    self.report({'ERROR'}, "Failed to enter edit mode")
                    return {'CANCELLED'}
                    
            # Force mesh update to ensure selection state is current
            force_update_mesh(obj)
            
            # Initialize BMesh - in Blender 4.x, this needs extra care
            debug_print("Setting up BMesh")
            self.mesh = obj.data
            self.bm = None
            
            # Special handling for selection mode - critically important for 4.3.2
            available_faces = []
            
            try:
                if scene.fractal_selected_only:
                    debug_print("SELECTION MODE ACTIVE - Using 4.3.2 special selection detection")
                    
                    # Use our special 4.3.2 selection detector
                    selected_faces, bm = get_selected_faces_431(obj)
                    
                    # Keep the BMesh if successful
                    if bm:
                        self.bm = bm
                    
                    if selected_faces:
                        available_faces = selected_faces
                        debug_print(f"Found {len(available_faces)} selected faces")
                    else:
                        self.report({'ERROR'}, "No faces are selected. Please select faces in Edit Mode first.")
                        self.cleanup(context)
                        return {'CANCELLED'}
                else:
                    # Regular mode - get all faces
                    if not self.bm:  # Only create a new BMesh if we didn't already get one
                        self.bm = bmesh.from_edit_mesh(self.mesh)
                        
                    ensure_bmesh_lookup_tables(self.bm)
                    available_faces = [f for f in self.bm.faces if validate_face(f)]
                    debug_print(f"Found {len(available_faces)} valid faces")
                    
                if not available_faces:
                    self.report({'ERROR'}, "No valid faces found in mesh")
                    self.cleanup(context)
                    return {'CANCELLED'}
                
            except Exception as e:
                debug_print(f"Error setting up faces: {e}")
                if DEBUG:
                    traceback.print_exc()
                self.report({'ERROR'}, f"Error setting up faces: {str(e)}")
                self.cleanup(context)
                return {'CANCELLED'}
            
            # Create processor
            debug_print("Creating processor")
            self.processor = FractalGeometryProcessor(context, self.bm)
            
            # Apply face limit
            num_faces = len(available_faces)
            face_limit = min(scene.fractal_face_limit, num_faces)
            debug_print(f"Processing {face_limit} faces out of {num_faces} available")
            
            # Select faces for processing
            try:
                if scene.fractal_selected_only:
                    # Use all selected faces up to limit
                    selected_faces = available_faces[:face_limit]
                else:
                    # Randomly sample from all faces - safely for Blender 4.x
                    if face_limit < num_faces:
                        # Simple sampling in 4.3.2
                        indices = list(range(num_faces))
                        random.shuffle(indices)
                        indices = indices[:face_limit]
                        selected_faces = [available_faces[i] for i in indices]
                    else:
                        selected_faces = available_faces
            except Exception as e:
                debug_print(f"Error selecting faces: {e}")
                if DEBUG:
                    traceback.print_exc()
                self.report({'ERROR'}, f"Error selecting faces: {str(e)}")
                self.cleanup(context)
                return {'CANCELLED'}
            
            # Log face details for debugging selection
            for i, face in enumerate(selected_faces[:5]):  # Just log first 5 faces
                debug_print(f"Face {i}: {len(face.verts)} verts, area: {face.calc_area():.6f}")
            
            # Create batches for processing
            batch_size = max(1, min(scene.fractal_batch_size, face_limit))
            self.face_batches = [
                selected_faces[i:i + batch_size]
                for i in range(0, len(selected_faces), batch_size)
            ]
            self.current_batch = 0
            
            debug_print(f"Created {len(self.face_batches)} batches with size {batch_size}")
            
            # Record start time for performance tracking
            self.start_time = time.time()
            
            # Start modal timer for processing - Blender 4.x compatible
            debug_print("Starting modal timer")
            wm = context.window_manager
            try:
                self._timer = wm.event_timer_add(0.1, window=context.window)
                wm.modal_handler_add(self)
            except Exception as e:
                debug_print(f"Error starting modal timer: {e}")
                if DEBUG:
                    traceback.print_exc()
                self.report({'ERROR'}, f"Error starting modal timer: {str(e)}")
                self.cleanup(context)
                return {'CANCELLED'}
            
            self.report({'INFO'}, f"Processing {face_limit} faces in {len(self.face_batches)} batches")
            debug_print(f"Started modal processing with {len(self.face_batches)} batches")
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            debug_print(f"Execute error: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            self.cleanup(context)
            return {'CANCELLED'}

    def cleanup(self, context):
        """Clean up when cancelled or on error - Blender 4.x compatible"""
        debug_print("Running cleanup")
        
        try:
            if hasattr(self, '_timer'):
                context.window_manager.event_timer_remove(self._timer)
                debug_print("Timer removed")
                
            if hasattr(self, 'bm') and self.bm:
                # Apply any changes made so far
                debug_print("Finalizing BMesh")
                try:
                    bmesh.update_edit_mesh(self.mesh)
                except Exception as e:
                    debug_print(f"Error updating edit mesh: {e}")
                
                try:
                    self.bm.free()
                    debug_print("BMesh freed")
                except Exception as e:
                    debug_print(f"Error freeing BMesh: {e}")
                
            context.window_manager.fractal_is_processing = False
            context.window_manager.fractal_should_cancel = False
            if hasattr(context.window_manager, 'fractal_progress'):
                context.window_manager.fractal_progress = 0.0
            
            # Restore original mode if needed - carefully in Blender 4.x
            if hasattr(self, 'original_mode'):
                debug_print(f"Restoring original mode: {self.original_mode}")
                try:
                    safe_mode_set(self.original_mode)
                except Exception as e:
                    debug_print(f"Failed to restore original mode: {e}")
                    
            # Force redraw of all 3D views for Blender 4.x
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
                    
        except Exception as e:
            debug_print(f"Error during cleanup: {e}")
            if DEBUG:
                traceback.print_exc()
            
    def cancel(self, context):
        """Clean up when cancelled by user"""
        debug_print("Operation cancelled by user")
        self.cleanup(context)
        self.report({'INFO'}, "Fractal generation cancelled")

    def finish(self, context):
        """Clean up when successfully completed"""
        debug_print("Operation finished successfully")
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        debug_print(f"Total processing time: {elapsed_time:.2f}s")
        
        try:
            # Get processor stats if available
            if hasattr(self, 'processor'):
                processed_count = getattr(self.processor, 'processed_count', 0)
                failed_count = getattr(self.processor, 'failed_count', 0)
                debug_print(f"Processed {processed_count} faces, failed {failed_count} faces")
                
            self.cleanup(context)
            self.report({'INFO'}, f"Fractal generation completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            debug_print(f"Error during finish: {e}")
            if DEBUG:
                traceback.print_exc()
            self.cleanup(context)

    
    def execute(self, context):
        debug_print("Cancel requested")
        context.window_manager.fractal_should_cancel = True
        return {'FINISHED'}

# ---- Property Registration ----

def register_properties():
    debug_print("Registering properties for Blender 4.3.2")
    
    # Processing state properties
    bpy.types.WindowManager.fractal_is_processing = BoolProperty(default=False)
    bpy.types.WindowManager.fractal_should_cancel = BoolProperty(default=False)
    bpy.types.WindowManager.fractal_progress = FloatProperty(default=0.0, min=0.0, max=100.0)
    
    # Pattern type selection
    bpy.types.Scene.fractal_type = EnumProperty(
        name="Fractal Type",
        description="Type of fractal pattern to generate",
        items=FRACTAL_TYPES,
        default='MANDELBROT'
    )
    
    # Julia set seed parameter
    bpy.types.Scene.fractal_julia_seed = FloatVectorProperty(
        name="Julia Seed",
        description="Seed values for Julia set (affects pattern)",
        default=(-0.8, 0.156),
        size=2,
        min=-2.0,
        max=2.0
    )
    
    # Extrusion direction control
    bpy.types.Scene.fractal_extrusion_direction = EnumProperty(
        name="Extrusion Direction",
        description="Direction to extrude faces",
        items=EXTRUSION_DIRECTIONS,
        default='NORMAL'
    )
    
    # Face selection properties
    bpy.types.Scene.fractal_selected_only = BoolProperty(
        name="Selected Faces Only",
        description="Apply fractal only to selected faces",
        default=False
    )
    
    bpy.types.Scene.fractal_face_limit = IntProperty(
        name="Face Limit",
        description="Maximum number of faces to process",
        default=100,
        min=1,
        max=5000
    )
    
    bpy.types.Scene.fractal_batch_size = IntProperty(
        name="Batch Size",
        description="Number of faces to process per batch",
        default=10,
        min=1,
        max=100
    )
    
    # Random seed
    bpy.types.Scene.fractal_random_seed = IntProperty(
        name="Random Seed",
        description="Seed value for random pattern generation (0 for time-based random)",
        default=0,
        min=0,
        max=9999
    )
    
    # Pattern settings
    bpy.types.Scene.fractal_iterations = IntProperty(
        name="Iterations",
        description="Number of fractal iterations (higher = more detailed but slower)",
        default=50,
        min=10,
        max=200
    )
    
    bpy.types.Scene.fractal_scale = FloatProperty(
        name="Scale",
        description="Scale of the fractal pattern",
        default=1.0,
        min=0.1,
        max=10.0
    )
    
    bpy.types.Scene.fractal_min_depth = FloatProperty(
        name="Minimum Depth",
        description="Minimum fractal value to trigger operations",
        default=0.1,
        min=0.0,
        max=1.0
    )
    
    # Geometry operation properties
    bpy.types.Scene.fractal_adapt_to_size = BoolProperty(
        name="Adapt to Face Size",
        description="Scale operations based on face size for more consistent results",
        default=True
    )
    
    bpy.types.Scene.fractal_inset_depth = FloatProperty(
        name="Initial Inset Depth",
        description="Depth of the first inset operation",
        default=0.3,
        min=0.01,
        max=1.0
    )
    
    bpy.types.Scene.fractal_secondary_inset = FloatProperty(
        name="Secondary Inset Depth",
        description="Depth of the secondary inset operations",
        default=0.2,
        min=0.01,
        max=1.0
    )
    
    bpy.types.Scene.fractal_extrusion_strength = FloatProperty(
        name="Extrusion Strength",
        description="Strength of the extrusion operations",
        default=0.5,
        min=0.01,
        max=2.0
    )
    
    bpy.types.Scene.fractal_recursion_chance = FloatProperty(
        name="Recursion Chance",
        description="Probability of additional inset/extrude operations (0-1)",
        default=0.7,
        min=0.0,
        max=1.0
    )

def unregister_properties():
    debug_print("Unregistering properties")
    
    try:
        # Remove all registered properties
        del bpy.types.WindowManager.fractal_is_processing
        del bpy.types.WindowManager.fractal_should_cancel
        del bpy.types.WindowManager.fractal_progress
        del bpy.types.Scene.fractal_type
        del bpy.types.Scene.fractal_julia_seed
        del bpy.types.Scene.fractal_extrusion_direction
        del bpy.types.Scene.fractal_selected_only
        del bpy.types.Scene.fractal_random_seed
        del bpy.types.Scene.fractal_face_limit
        del bpy.types.Scene.fractal_batch_size
        del bpy.types.Scene.fractal_iterations
        del bpy.types.Scene.fractal_scale
        del bpy.types.Scene.fractal_min_depth
        del bpy.types.Scene.fractal_inset_depth
        del bpy.types.Scene.fractal_secondary_inset
        del bpy.types.Scene.fractal_extrusion_strength
        del bpy.types.Scene.fractal_recursion_chance
        del bpy.types.Scene.fractal_adapt_to_size
    except:
        if DEBUG:
            traceback.print_exc()
        debug_print("Error during property unregistration")

# ---- Registration ----

classes = (
    FRACTAL_PT_main_panel,
    MESH_OT_fractal_generate,
    MESH_OT_fractal_cancel,
    MESH_OT_fractal_debug,
)

def register():
    debug_print("Registering addon for Blender 4.3.2")
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            debug_print(f"Registered class: {cls.__name__}")
        except Exception as e:
            debug_print(f"Error registering {cls.__name__}: {e}")
            if DEBUG:
                traceback.print_exc()
    register_properties()
    debug_print("Addon registration complete")

def unregister():
    debug_print("Unregistering addon")
    unregister_properties()
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            debug_print(f"Unregistered class: {cls.__name__}")
        except Exception as e:
            debug_print(f"Error unregistering {cls.__name__}: {e}")
            if DEBUG:
                traceback.print_exc()
    debug_print("Addon unregistration complete")

if __name__ == "__main__":
    register()
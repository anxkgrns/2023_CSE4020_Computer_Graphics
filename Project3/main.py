from OpenGL.GL import *
from glfw.GLFW import *
import glm 
import numpy as np
import os


width = 1200
height = 800
pm = 1

mode = 0

g_cam_center = glm.vec3(0,0,0)

#g_cam_location = glm.vec3(3*0.8,3*0.8,-3*1.5)

sc = 3
g_cam_location = glm.vec3(sc*0.8,sc*0.8,-sc*1.5)
g_cam_location_norm = glm.normalize(g_cam_location)

azimuth = np.arctan(g_cam_location_norm.z/g_cam_location_norm.x)
elevation = np.arcsin(g_cam_location_norm.y)

g_cam_w = glm.vec3(0,0,0)
g_cam_u = glm.vec3(0,0,0)
g_cam_v = glm.vec3(0,0,0)
g_cam_up = glm.vec3(0,1,0)

size = 10
size_t = size
firstMouse = 0
lastX = 0
lastY = 0

bvh = 0

path = None
path_before = None

toggle_solid_mode = 0

line_render = 1
box_render = 0
animation_mode = 0
before_frame_time = 0

frames = 0
frame_time = 0
frames_data = []

points = []
J = []
J_num = []
skeleton = []
L = []
L_P_list = []
depth = []

now_frame = 0 

vao_lines = []

new_vao_lines = []
new_vao_boxes = []
vao_parents = []
scc = 1

first = 2

first_grid = 0
first_floor = 0

s_vertices = glm.array(glm.float32,0,0,0)

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

g_vertex_shader_src_light = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M))) * vin_normal);
}
'''

g_fragment_shader_src_light = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 material_color;
uniform vec3 view_pos;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,3,-4);
    vec3 light_color = vec3(1,1,1);
    //vec3 material_color = vec3(1,0,0);
    float material_shininess = 50.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 0.9);
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def key_callback(window, key, scancode, action, mods):
    global mode, box_render, line_render, animation_mode
    global first_grid, first_floor
    if key==GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                mode = (mode+1)%2
            if key==GLFW_KEY_1:
                line_render = 1
                box_render = 0
                first_grid = 1
            if key==GLFW_KEY_2:
                box_render = 1
                line_render = 0
                first_floor = 1
            if key==GLFW_KEY_SPACE:
                #animation_mode = (animation_mode+1)%2
                animation_mode = 1

def camera_far(p1,p2):
    return np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

def mouse_zoom_callback(window, xoffset, yoffset):
    global g_cam_location, g_cam_w, mode
    if(mode == 0):
        senitive = 0.05
        if((camera_far(g_cam_center, g_cam_location) > 1 ) or yoffset < 0):
            g_cam_location = g_cam_location - g_cam_w*yoffset*senitive
    else:
        senitive = 0.05
        if((camera_far(g_cam_center, g_cam_location) > 1 ) or yoffset < 0):
            g_cam_location = g_cam_location - g_cam_w*yoffset*senitive
    
def mouse_button_callback(window, button, action, mods):
    global firstMouse,g_cam_w,g_cam_u,g_cam_v,g_cam_up
    if action == GLFW_PRESS or action == GLFW_REPEAT:
        if button == GLFW_MOUSE_BUTTON_LEFT:
            firstMouse = 1
        elif button == GLFW_MOUSE_BUTTON_RIGHT:
            firstMouse = 2
    else:
        
        firstMouse = 0

def mouse_callback(window, xpos, ypos):
    global  g_cam_center, g_cam_location, firstMouse, lastX, lastY,g_cam_w,g_cam_up,g_cam_v,pm, azimuth, elevation, g_cam_u
    length = camera_far(g_cam_center, g_cam_location)
    if firstMouse == 1:
        lastX = xpos
        lastY = ypos
        firstMouse = 3
        
    elif firstMouse == 2:
        lastX = xpos
        lastY = ypos
        firstMouse = 4

    elif firstMouse == 3:
        xoffset = lastX - xpos
        yoffset = ypos - lastY
        lastX = xpos
        lastY = ypos
        sensitive = 0.0025

        #azimuth = np.arctan(g_cam_w.z/g_cam_w.x)
        #elevation = np.arcsin(g_cam_w.y)
        azimuth = (azimuth - pm*xoffset*sensitive)
        elevation = elevation + pm*yoffset*sensitive
        
        if(elevation > np.pi/2):
            pm *= -1
            g_cam_up = glm.vec3(0, 1*pm, 0)
            elevation = np.pi - elevation
            azimuth = azimuth - np.pi

        elif(elevation < -np.pi/2):
            pm *= -1
            g_cam_up = glm.vec3(0, 1*pm, 0)
            elevation = -np.pi - elevation
            azimuth = azimuth + np.pi      

        g_cam_w = glm.normalize(glm.vec3(np.cos(azimuth)*np.cos(elevation), np.sin(elevation), np.sin(azimuth)*np.cos(elevation)))
        #g_cam_w = glm.normalize(g_cam_u*xoffset*sensitive + g_cam_v*yoffset*sensitive + g_cam_w)
        g_cam_location = g_cam_w*length + g_cam_center



        #print(pm, azimuth, elevation)

    elif firstMouse == 4:
        xoffset = lastX - xpos
        yoffset = ypos - lastY
        lastX = xpos
        lastY = ypos
        sensitive = 0.0005*length

        g_cam_center = g_cam_center + g_cam_u*xoffset*sensitive + g_cam_v*yoffset*sensitive
        g_cam_location = g_cam_location + g_cam_u*xoffset*sensitive + g_cam_v*yoffset*sensitive
        
def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32, 
        # position      normal
        -1 ,  1 ,  1 ,  0, 0, 1, # v0
         1 , -1 ,  1 ,  0, 0, 1, # v2
         1 ,  1 ,  1 ,  0, 0, 1, # v1

        -1 ,  1 ,  1 ,  0, 0, 1, # v0
        -1 , -1 ,  1 ,  0, 0, 1, # v3
         1 , -1 ,  1 ,  0, 0, 1, # v2

        -1 ,  1 , -1 ,  0, 0, 1, # v4
         1 ,  1 , -1 ,  0, 0, 1, # v5
         1 , -1 , -1 ,  0, 0, 1, # v6

        -1 ,  1 , -1 ,  0, 0, 1, # v4
         1 , -1 , -1 ,  0, 0, 1, # v6
        -1 , -1 , -1 ,  0, 0, 1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 , -1 ,  1 ,  0, 1, 0, # v3
         1 , -1 , -1 ,  0, 1, 0, # v6
         1 , -1 ,  1 ,  0, 1, 0, # v2

        -1 , -1 ,  1 ,  0, 1, 0, # v3
        -1 , -1 , -1 ,  0, 1, 0, # v7
         1 , -1 , -1 ,  0, 1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 ,  1 ,  1, 0, 0, # v2
         1 , -1 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 ,  1, 0, 0, # v0
        -1 , -1 , -1 ,  1, 0, 0, # v7
        -1 , -1 ,  1 ,  1, 0, 0, # v3

        -1 ,  1 ,  1 ,  1, 0, 0, # v0
        -1 ,  1 , -1 ,  1, 0, 0, # v4
        -1 , -1 , -1 ,  1, 0, 0, # v7
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    # vertices = glm.array(glm.float32,
    #     # position        # color
    #      -size_t, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
    #      size_t, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
    #      0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
    #      0.0, size_t, 0.0,  0.0, 1.0, 0.0, # y-axis end 
    #      0.0, 0.0, -size_t,  0.0, 0.0, 1.0, # z-axis start
    #      0.0, 0.0, size_t,  0.0, 0.0, 1.0, # z-axis end 
    # )

    vertices = glm.array(glm.float32,
        # position        # color
         -1, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         1, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 1, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -1,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 1,  0.0, 0.0, 1.0, # z-axis end 
    )


    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid_x():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         size_t, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis start
         -size_t, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid_z():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, size_t,  0.5, 0.5, 0.5, # x-axis start
         0.0, 0.0, -size_t,  0.5, 0.5, 0.5, # x-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid_center_x():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         size_t, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis start
         1.0, 0.0, 0.0, 0.5, 0.5, 0.5, # x-axis start

         -size_t, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis end 
         -1.0, 0.0, 0.0, 0.5, 0.5, 0.5, # x-axis start
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid_center_z():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, size_t,  0.5, 0.5, 0.5, # x-axis start
         0.0, 0.0, 1.0,  0.5, 0.5, 0.5, 

         0.0, 0.0, -size_t,  0.5, 0.5, 0.5, # x-axis end 
         0.0, 0.0, -1.0,  0.5, 0.5, 0.5,  
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_floor():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  0, 1, 0, 
         0.0, 0.0, 1.0,  0, 1, 0, 
         1.0, 0.0, 1.0,  0, 1, 0, 
         
         0.0, 0.0, 0.0,  0, 1, 0, 
         1.0, 0.0, 0.0,  0, 1, 0, 
         1.0, 0.0, 1.0,  0, 1, 0, 

    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_floor(vao_floor, MVP, MVP_loc, M, M_loc, color_loc_light, view_pos_loc):
    glBindVertexArray(vao_floor)
    temp_size = 1
    for i in range(0,temp_size*4):
        i -= temp_size*2
        for j in range(0,temp_size*4):
            j -= temp_size*2
            if ((i+j)%2 == 0):
                MVP_floor = MVP * glm.translate(glm.vec3(j, 0, i))
                glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_floor))
                glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
                glUniform3f(view_pos_loc, g_cam_location.x, g_cam_location.y, g_cam_location.z)
                glUniform3f(color_loc_light, 0.7, 0.7, 0.7)
                glDrawArrays(GL_TRIANGLES, 0, 24)
            else:
                MVP_floor = MVP * glm.translate(glm.vec3(j, 0, i))
                glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_floor))
                glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
                glUniform3f(view_pos_loc, g_cam_location.x, g_cam_location.y, g_cam_location.z)
                glUniform3f(color_loc_light, 0.4, 0.4, 0.4)
                glDrawArrays(GL_TRIANGLES, 0, 24)

def draw_grid(vao_x, vao_z, vao_cenx, vao_cenz, MVP, MVP_loc):
    glBindVertexArray(vao_x)
    for i in range(1,size*2):
        i -= size
        if(i != 0):
            MVP_grid_x = MVP * glm.translate(glm.vec3(0, 0, i))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_x))
            glDrawArrays(GL_LINES, 0, 2)

    glBindVertexArray(vao_cenx)
    MVP_grid_cenx = MVP * glm.translate(glm.vec3(0, 0, 0))
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_cenx))
    glDrawArrays(GL_LINES, 0, 4)
    
    glBindVertexArray(vao_z)
    for j in range(1,size*2):
        j -= size
        if(j != 0):
            MVP_grid_z = MVP * glm.translate(glm.vec3(j, 0, 0))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_z))
            glDrawArrays(GL_LINES, 0, 2)
    
    glBindVertexArray(vao_cenz)
    MVP_grid_cenz = MVP * glm.translate(glm.vec3(0, 0, 0))
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_cenz))
    glDrawArrays(GL_LINES, 0, 4)

def framebuffer_size_callback(window, x, y):
    glViewport(0, 0, x, (int)(x*height/width))



depth_L = []
depth_Ls_list = []
skeleton_L = []
rel_skeleton = []
M_line = []
scales = []
Rys = []


def drop_callback(window, paths):
    global single_mesh_index, s_vertices, path, path_before, bvh, vao_lines, frame_time, frames, frames_data, points, J, J_num, L, skeleton,L_P_list, depth, animation_mode, now_frame
    global new_vao_lines, new_vao_boxes, vao_parents
    global first, M_line
    global scales, Rys, scc
    path = paths
    animation_mode = 0
    now_frame = 0


    ROOT_offset = []
    root = 0
    skeleton = []
    M_line = []
    L = [] #link translation - skeleton
    J = [] #joint rotation - motion
    J_num =[] # joint rotation's elements number
    end = 0
    P_nums = [] # end site M numbers
    p_num = -1
    P = [] # end site offsets
    depth = [] # point들 사이의 깊이 연결
    points = [] 

    # motion
    motion = 0
    frames = 0
    frame_time = 0
    frames_data = []

    L_P_list = []

    new_vao_lines = []
    new_vao_boxes = []
    vao_parents = []

    Rys = []
    scales = []

    max = 0
    min = 0
    scc = 1


    
    with open(paths[0], 'r') as file:
        file_name = str(paths[0]).split('\\')[-1]
        #file_name = os.path.basename(paths[0])
        print("File name :",file_name)
        bvh = 1
        temp = 0 # L이면 1 end P이면 0
        single_mesh_index = 0
        s_vertices = None
        #scc =0.01
        lines = file.readline()
        while lines:
            #print(lines)
            elements = lines.split()
            if len(elements) == 0:
                lines = file.readline()
                continue
            elif elements[0] == "HIEARCHY":
                lines = file.readline()
                continue
                #start
            elif elements[0] == "ROOT":
                root = 1
                skeleton.append(elements[1])
            elif elements[0] == "JOINT":
                skeleton.append(elements[1])
            elif elements[0] == "OFFSET":
                single_off = [float(elements[1]), float(elements[2]), float(elements[3])]#[float(elements[1])*scc, float(elements[2])*scc, float(elements[3])*scc]
                depth.append(p_num+1)
                points.append(single_off)

                if root == 1:
                    temp = 2
                    ROOT_offset.append(single_off)
                    root = 0
                elif root == 0 and end == 0:
                    temp = 1
                    L.append(single_off)
                elif end == 1:
                    temp = 0
                    P.append(single_off)
                    end = 0
                L_P_list.append(temp)

                if float(elements[1]) > max:
                    max = float(elements[1])
                elif float(elements[1]) < min:
                    min = float(elements[1])
                if float(elements[2]) > max:
                    max = float(elements[2])
                elif float(elements[2]) < min:
                    min = float(elements[2])
                if float(elements[3]) > max:
                    max = float(elements[3])
                elif float(elements[3]) < min:
                    min = float(elements[3])
                
            elif elements[0] == "CHANNELS":
                if(elements[1] == "3"):
                    J_num.append(3)
                    J.append([elements[2][0],elements[3][0],elements[4][0]])
                elif(elements[1] == "6"):
                    J_num.append(6)
                    J.append([elements[2][0],elements[3][0],elements[4][0],elements[5][0],elements[6][0],elements[7][0]])
            elif elements[0] == "End" and elements[1] == "Site":
                end = 1
            elif elements[0] == '{':
                p_num += 1
                lines = file.readline()
                continue
            elif elements[0] == '}':
                p_num -= 1
                lines = file.readline()
                continue
            elif elements[0] == "MOTION":
                motion = 1
                lines = file.readline()
                continue
            elif elements[0] == "Frames:":
                frames = int(elements[1])
            elif elements[0] == "Frame" and elements[1] == "Time:":
                frame_time = float(elements[2])
            elif motion == 1:
                single_frame = []
                for i in range(len(elements)):
                    single_frame.append(float(elements[i])) # extend ?
                frames_data.append(single_frame)

            lines = file.readline()
    print("Numbers of frames :", frames)
    # formatted_fps = "%.6f" % float(1/frame_time)
    # print("FPS(1/FrameTime) :", formatted_fps)
    print("FPS(1/FrameTime) :", float(1/frame_time))
    print("Number of joints :", len(skeleton))
    print("List of all joint names :",end=' ')
    for i in range (len(skeleton)):
        print(skeleton[i], end = ' ')
    print("")
    print("")
    
    
    first = 1
    # max의 절댓값과 min의 절댓값 중 더 큰 것의 자릿수를 보고 scale을 결정


    if(abs(max) > abs(min)):
        x = abs(max)
        while(x > 3):
            x = x*0.1
            scc = float(scc*0.1)
    else:
        x = abs(min)
        while(x > 3):
            x = x*0.1
            scc = float(scc*0.1)
    


    for i in range(len(points)):
        for j in range(3):
            points[i][j] = points[i][j]*scc
    


    # points = [points[i]*0.01 for i in range(len(points))]
    # frames_data = [frames_data[i] for i in range(len(frames_data))]
    # print(L)
    # print(J)
    # print(depth)
    # print("points",points)
    # make vertices
    #vertices_n = np.array(v_vn, dtype=np.float32)
    #s_vertices = glm.array(vertices_n)
    #vao_single_mesh = prepare_single_mesh()
    #print("result",s_vertices)

# num 번째 rotate matrix를 계산함
def J_rotate(num):
    if(animation_mode == 0):
        return glm.mat4()
    sum = 0 
    for i in range(len(J_num)):
        if(i == num):
            break
        else:
            sum += J_num[i]

    mat = []

    if(J_num[num] == 3):
        for i in range(3):
            rad = glm.radians(frames_data[now_frame][sum+i])
            if(J[num][i] == 'X'):
                mat.append(glm.rotate(rad, (1,0,0)))
            elif(J[num][i] == 'Y'):
                mat.append(glm.rotate(rad, (0,1,0)))
            elif(J[num][i] == 'Z'):
                mat.append(glm.rotate(rad, (0,0,1)))
        J_mat = glm.mat4(mat[0] * mat[1] * mat[2])

    elif(J_num[num] == 6):
        for i in range(3):
            if(J[num][i] == 'X'):
                x = (frames_data[now_frame][sum+i])
            elif(J[num][i] == 'Y'):
                y = (frames_data[now_frame][sum+i])
            elif(J[num][i] == 'Z'):
                z = (frames_data[now_frame][sum+i])
        for j in range(3,6):
            rad = glm.radians(frames_data[now_frame][sum+j])
            if(J[num][j] == 'X'):
                mat.append(glm.rotate(rad, (1,0,0)))
            elif(J[num][j] == 'Y'):
                mat.append(glm.rotate(rad, (0,1,0)))
            elif(J[num][j] == 'Z'):
                mat.append(glm.rotate(rad, (0,0,1)))
        J_mat = glm.translate(glm.vec3(x*scc,y*scc,z*scc)) * glm.mat4(mat[0] * mat[1] * mat[2])
    
    if(animation_mode == 1):
        return J_mat


def setting_frame_line(frame_num,vao_line,VP,MVP_loc):
    global vao_lines
    global now_frame

    global new_vao_lines, vao_parents
    global first
    global depth_L, depth_Ls_list, skeleton_L, rel_skeleton
    global M_line, scales, Rys
    now_frame = frame_num
    if(first == 1):
        depth_L = []
        depth_Ls_list = []
        skeleton_L = []
        for i in range(len(depth)):
            if(L_P_list[i] == 1 or L_P_list[i] == 2):
                depth_L.append(depth[i])
                depth_Ls_list.append(i)
        for i in range(len(points)):
            if(L_P_list[i] == 1 or L_P_list[i] == 2):
                skeleton_L.append(points[i])

    # print("skeleton_L",skeleton_L)
    # print("len_skelteton_L",len(skeleton_L))
    # print("depth_L",depth_L)
    # print("depth_Ls_list",depth_Ls_list)
    # print("len_L_P_list",len(L_P_list))
    # print("L_P_LIST",L_P_list)
    M_line = []
    for i in range(len(skeleton)):
        if(i == 0):
            #M.append(glm.mat4())
            M_line.append(glm.mat4()*J_rotate(0)*glm.translate(glm.vec3(skeleton_L[0][0],skeleton_L[0][1],skeleton_L[0][2])))
            # print("M",M)
        x = depth_L[i]
        if(i != 0):
            for j in range(i,-1,-1):
                if(depth_L[j] == x-1):
                    M_line.append(M_line[j] * glm.translate(glm.vec3(skeleton_L[i][0],skeleton_L[i][1],skeleton_L[i][2]))* J_rotate(i) )
                    break
    
    #print(M)
    # print("M_len",len(M))
    # print("depth",depth)
    # print("depth_L",depth_L)
    # print("depth_L_len",len(depth_L))
    # print("rel_points_len", len(rel_points))
    #vao_lines.clear()

    # 선언은 오직 한번만
    if(new_vao_lines == []):
        vao_parents = []
        rel_points = points[:]
        for i in range(len(rel_points)):
            if(i == 0):
                continue
            point2 = rel_points[i] #glm.vec3(M[i]*glm.vec4(rel_points[i],1.0))
            x = depth[i]
            for j in range(i,-1,-1):
                if(depth[j] == x-1):
                    point1 = rel_points[j] #glm.vec3(M[j]*glm.vec4(rel_points[j],1.0))
                    vao_parents.append(j)
                    break
            #vao_lines.append(prepare_vao_line(point1,point2))
            new_vao_lines.append("1")#prepare_vao_line([0,0,0],point2))


    # print("vao_lines",len(new_vao_lines))
    # print("vao_parents",vao_parents)
    if(Rys == []):
        rel_points = points[:]
        scales = []
        for i in range(len(new_vao_lines)):
            scale = camera_far(glm.vec3(0,0,0),glm.vec3(rel_points[i+1]))
            scales.append(scale)
            if(rel_points[i+1][0] == 0 and rel_points[i+1][2] == 0):
                Ry = glm.mat4()

            elif(rel_points[i+1][0] != 0 or rel_points[i+1][2] != 0):
                if (rel_points[i+1][1]/scale) > 1:
                    rad_y = glm.radians(0)
                elif (rel_points[i+1][1]/scale) < -1:
                    rad_y = glm.radians(180)
                else :
                    rad_y = np.arccos(rel_points[i+1][1]/scale)
                axis = glm.cross(glm.vec3(0,1,0),glm.vec3(rel_points[i+1]))
                Ry = glm.rotate(rad_y, axis)
            #else:
            Rys.append(Ry)

    if(first == 1):
        rel_skeleton = []
        for i in range(len(vao_parents)):
            for j in range(len(depth_Ls_list)):
                if(vao_parents[i]==depth_Ls_list[j]):
                    rel_skeleton.append(j)
                    break
    # print("rel_skeleton",rel_skeleton)

    for i in range(len(new_vao_lines)):
        new_MVP = VP * M_line[rel_skeleton[i]] * Rys[i] * glm.scale((0.02,scales[i],0.02))
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(new_MVP))
        glBindVertexArray(vao_line)
        glDrawArrays(GL_LINES, 0, 2)
        
    if(first == 1):
        first = 0 
    

def setting_frame_box(frame_num,vao_box,VP,MVP_loc,M_loc):
    global vao_lines
    global now_frame
    global new_vao_boxes, vao_parents
    
    global first
    global depth_L, depth_Ls_list, skeleton_L, rel_skeleton
    global scales, Rys
    global M_line

    now_frame = frame_num
    

    if(first == 1):
        depth_L = []
        depth_Ls_list = []
        skeleton_L = []
        for i in range(len(depth)):
            if(L_P_list[i] == 1 or L_P_list[i] == 2):
                depth_L.append(depth[i])
                depth_Ls_list.append(i)
        for i in range(len(points)):
            if(L_P_list[i] == 1 or L_P_list[i] == 2):
                skeleton_L.append(points[i])

    # print("skeleton_L",skeleton_L)
    # print("len_skelteton_L",len(skeleton_L))
    # print("depth_L",depth_L)
    # print("depth_Ls_list",depth_Ls_list)
    # print("len_L_P_list",len(L_P_list))
    # print("L_P_LIST",L_P_list)
    M_line = []
    for i in range(len(skeleton)):
        if(i == 0):
            #M.append(glm.mat4())
            M_line.append(glm.mat4()*J_rotate(0)*glm.translate(glm.vec3(skeleton_L[0][0],skeleton_L[0][1],skeleton_L[0][2])))
            # print("M",M)
        x = depth_L[i]
        if(i != 0):
            for j in range(i,-1,-1):
                if(depth_L[j] == x-1):
                    # print("j",j)
                    # print("i",i)
                    M_line.append(M_line[j] * glm.translate(glm.vec3(skeleton_L[i][0],skeleton_L[i][1],skeleton_L[i][2]))* J_rotate(i) )
                    break
    
    #print(M)
    # print("M_len",len(M))
    # print("depth",depth)
    # print("depth_L",depth_L)
    # print("depth_L_len",len(depth_L))
    # print("rel_points_len", len(rel_points))
    #vao_lines.clear()
    if(new_vao_boxes == []):
        rel_points = points[:]
        for i in range(len(rel_points)):
            if(i == 0):
                continue
            point2 = rel_points[i] #glm.vec3(M[i]*glm.vec4(rel_points[i],1.0))
            x = depth[i]
            for j in range(i,-1,-1):
                if(depth[j] == x-1):
                    point1 = rel_points[j] #glm.vec3(M[j]*glm.vec4(rel_points[j],1.0))
                    vao_parents.append(j)
                    break
            #vao_lines.append(prepare_vao_line(point1,point2))
            new_vao_boxes.append("1")#prepare_vao_box())
    # print("vao_boxes",len(new_vao_boxes))
    # print("vao_parents",vao_parents)
    if(first == 1):
        rel_skeleton = []
        for i in range(len(vao_parents)):
            for j in range(len(depth_Ls_list)):
                if(vao_parents[i]==depth_Ls_list[j]):
                    rel_skeleton.append(j)
                    break
    # print("rel_skeleton",rel_skeleton)

    if(Rys == []):
        rel_points = points[:]
        scales = []
        for i in range(len(new_vao_boxes)):
            scale = camera_far(glm.vec3(0,0,0),glm.vec3(rel_points[i+1]))
            scales.append(scale)
            if(rel_points[i+1][0] == 0 and rel_points[i+1][2] == 0):
                Ry = glm.mat4()

            elif(rel_points[i+1][0] != 0 or rel_points[i+1][2] != 0):
                if (rel_points[i+1][1]/scale) > 1:
                    rad_y = glm.radians(0)
                elif (rel_points[i+1][1]/scale) < -1:
                    rad_y = glm.radians(180)
                else :
                    rad_y = np.arccos(rel_points[i+1][1]/scale)
                axis = glm.cross(glm.vec3(0,1,0),glm.vec3(rel_points[i+1]))
                Ry = glm.rotate(rad_y, axis)
            #else:

            Rys.append(Ry)



    for i in range(len(new_vao_boxes)):
        # scale = camera_far(glm.vec3(0,0,0),glm.vec3(rel_points[i+1]))
        # if(rel_points[i+1][0] != 0 or rel_points[i+1][2] != 0):
        #     rad_y = np.arccos(rel_points[i+1][1]/scale)
        #     #print("rad_y",rad_y)
        #     axis = glm.cross(glm.vec3(0,1,0),glm.vec3(rel_points[i+1]))
        #     Ry = glm.rotate(rad_y, axis)
        #     #print("Ry",Ry)
        #     new_M = M_line[rel_skeleton[i]] *Ry* glm.scale((0.02,scale,0.02))
        # else:
        #     new_M = M_line[rel_skeleton[i]] * glm.scale((0.02,scale,0.02))
        new_M = M_line[rel_skeleton[i]] * Rys[i] * glm.scale((0.02,scales[i],0.02))
        new_MVP = VP*new_M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(new_MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(new_M))

        #MVP = VP
        #glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glBindVertexArray(vao_box)
        glDrawArrays(GL_TRIANGLES, 0, 36)
    if(first == 1):
        first = 0 


def prepare_vao_box():
    # prepare vertex data (in main memory)
    # 6 vertices for 2 triangles
    vertices = glm.array(glm.float32,
        # position      # normal
        -1 ,  1 ,  1 ,  0, 0, 1, # v0
         1 ,  0 ,  1 ,  0, 0, 1, # v2
         1 ,  1 ,  1 ,  0, 0, 1, # v1

        -1 ,  1 ,  1 ,  0, 0, 1, # v0
        -1 ,  0 ,  1 ,  0, 0, 1, # v3
         1 ,  0 ,  1 ,  0, 0, 1, # v2

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  1 , -1 ,  0, 0,-1, # v5
         1 ,  0 , -1 ,  0, 0,-1, # v6

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  0 , -1 ,  0, 0,-1, # v6
        -1 ,  0 , -1 ,  0, 0,-1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 ,  0 ,  1 ,  0,-1, 0, # v3
         1 ,  0 , -1 ,  0,-1, 0, # v6
         1 ,  0 ,  1 ,  0,-1, 0, # v2

        -1 ,  0 ,  1 ,  0,-1, 0, # v3
        -1 ,  0 , -1 ,  0,-1, 0, # v7
         1 ,  0 , -1 ,  0,-1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 ,  0 ,  1 ,  1, 0, 0, # v2
         1 ,  0 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 ,  0 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  0 , -1 , -1, 0, 0, # v7
        -1 ,  0 ,  1 , -1, 0, 0, # v3

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  1 , -1 , -1, 0, 0, # v4
        -1 ,  0 , -1 , -1, 0, 0, # v7

    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)


    return VAO

def prepare_vao_lines():
    vertices = glm.array(glm.float32,
        # position      # normal
         0 ,  1 ,  0 ,  0.29, 0.43, 0.98, # v0
         0 ,  0 ,  0 ,  0.29, 0.43, 0.98, # v2
    )
    # print("result",s_vertices)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_line(point1,point2):
    points = []
    pointx = [element for element in point1]
    pointx.append(0.29)
    pointx.append(0.43)
    pointx.append(0.98)
    points.extend(pointx)
    # print(pointx)
    pointy = [element for element in point2]
    pointy.append(0.29)
    pointy.append(0.43)
    pointy.append(0.98)
    points.extend(pointy)
    # print(pointy)
    vertices_n = np.array(points, dtype=np.float32)
    s_vertices = glm.array(glm.float32,0,0,0,0,0,0)
    s_vertices = glm.array(vertices_n)
    # print("result",s_vertices)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, s_vertices.nbytes, s_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_single_mesh():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    global s_vertices

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, s_vertices.nbytes, s_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO
     
def animation_mesh(vertices):
    # prepare vertex data (in main memory)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_node(vao, index, node, VP, MVP_loc, M_loc, color_loc):
    M = node.get_global_transform() * node.get_shape_transform()
    MVP = VP * M
    color = node.get_color()

    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, index)

class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def main():

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(width, height, '2021076308', None, None)

    # render


    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetDropCallback(window, drop_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetCursorPosCallback(window, mouse_callback)
    glfwSetScrollCallback(window, mouse_zoom_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    shader_program_light = load_shaders(g_vertex_shader_src_light, g_fragment_shader_src_light)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    MVP_loc_light = glGetUniformLocation(shader_program_light, 'MVP')
    M_loc_light = glGetUniformLocation(shader_program_light, 'M')
    view_pos_loc_light = glGetUniformLocation(shader_program_light, 'view_pos')
    color_loc_light = glGetUniformLocation(shader_program_light, 'material_color')
    
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid_x = prepare_vao_grid_x()
    vao_grid_z = prepare_vao_grid_z()
    vao_grid_cenx = prepare_vao_grid_center_x()
    vao_grid_cenz = prepare_vao_grid_center_z()

    vao_floor = prepare_vao_floor()
    vao_triangle = prepare_vao_triangle() # 기본 상태 
    vao_line = prepare_vao_lines()
    vao_box = prepare_vao_box()

    # create a hirarchical model - Node(parent, link_transform_from_parent, shape_transform, color)
    
    global first_grid, first_floor

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        global g_cam_w,g_cam_u,g_cam_v,g_cam_location,g_cam_center,g_cam_up, now_frame, before_frame_time
        
        g_cam_w = glm.normalize(g_cam_location - g_cam_center) # 카메라 뒷방향
        g_cam_u = glm.normalize(glm.cross(g_cam_up,g_cam_w)) # 카메라 오른쪽방향
        g_cam_v = glm.normalize(glm.cross(g_cam_w, g_cam_u)) # 카메라 윗방향

        # enable depth test (we'll see details later)
        # 3차원 구현을 위해서는 거의 필요하다. 카메라에 가까운거는 더 앞에 그려지고, 멀리 있는 것은 뒤에 그려진다. 
        # 물체 뒤에 있는거는 가려지고 앞에 있는 것은 보인다.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        # use orthogonal projection (we'll see details later)
        
        if(mode == 0):
            P = glm.perspective(glm.radians(45.0), width/height, 0.1, 10000.0)

        if(mode == 1):
            #P = glm.ortho(-the_for_ortho*width/height,the_for_ortho*width/height,-the_for_ortho,the_for_ortho,-10000,10000)
            P = glm.ortho(-size_t/4*width/height,size_t/4*width/height,-size_t/4,size_t/4,-10000,10000)

        if(toggle_solid_mode == 0):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif(toggle_solid_mode == 1):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        # view matrix

        V = glm.lookAt(g_cam_location, g_cam_center, g_cam_up)

        # current frame: P*V*I (now this is the world frame)
        M = glm.mat4()

        # current frame: P*V*M
        MVP = P*V*M
        VP = P*V
        
        # draw triangle w.r.t. the current frame
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # draw reference grid face
        draw_grid(vao_grid_x,vao_grid_z,vao_grid_cenx,vao_grid_cenz,MVP,MVP_loc)
            

        # draw grid

        t = glfwGetTime()


        if(bvh == 0):
            # glUseProgram(shader_program)
            # draw_grid(vao_grid_x,vao_grid_z,vao_grid_cenx,vao_grid_cenz,MVP,MVP_loc)
            # glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_triangle)
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glDrawArrays(GL_TRIANGLES, 0, 36)

        # draw animation bvh
        elif(bvh == 1):
            if(line_render == 1):
                # glUseProgram(shader_program)
                # draw_grid(vao_grid_x,vao_grid_z,vao_grid_cenx,vao_grid_cenz,MVP,MVP_loc)
                    
                if(animation_mode == 0):
                    setting_frame_line(now_frame,vao_line,VP,MVP_loc)
                elif(animation_mode == 1):
                    if(now_frame < frames-1):
                        #print("now_frame",now_frame)
                        if(glfwGetTime() - before_frame_time > frame_time):
                            now_frame += 1
                            before_frame_time = glfwGetTime()
                        setting_frame_line(now_frame,vao_line,VP,MVP_loc)

                    elif(now_frame == frames-1):
                        now_frame = 0
                        before_frame_time = glfwGetTime()
                        setting_frame_line(now_frame,vao_line,VP,MVP_loc)
            if(box_render == 1):
                glUseProgram(shader_program_light)
                #draw_floor(vao_floor, MVP, MVP_loc_light, M, M_loc_light, color_loc_light, view_pos_loc_light)
                
                glUniform3f(view_pos_loc_light, g_cam_location.x, g_cam_location.y, g_cam_location.z)
                glUniform3f(color_loc_light, 0.29, 0.43, 0.8)
                if(animation_mode == 0):
                    setting_frame_box(now_frame,vao_box,VP,MVP_loc_light,M_loc_light)
                elif(animation_mode == 1):
                    if(now_frame < frames-1):
                        #print("now_frame",now_frame)
                        if(glfwGetTime() - before_frame_time > frame_time):
                            now_frame += 1
                            before_frame_time = glfwGetTime()
                        setting_frame_box(now_frame,vao_box,VP,MVP_loc_light,M_loc_light)

                    elif(now_frame == frames-1):
                        now_frame = 0
                        before_frame_time = glfwGetTime()
                        setting_frame_box(now_frame,vao_box,VP,MVP_loc_light,M_loc_light)
                
                
                # glBindVertexArray(vao_box)
                # M = M*glm.scale((0.1,2,0.1))
                # MVP = MVP*glm.scale((0.1,2,0.1))
                # glUniformMatrix4fv(MVP_loc_light, 1, GL_FALSE, glm.value_ptr(MVP))
                # glUniformMatrix4fv(M_loc_light, 1, GL_FALSE, glm.value_ptr(M))
                # glUniform3f(view_pos_loc_light, g_cam_location.x, g_cam_location.y, g_cam_location.z)
                # glUniform3f(color_loc_light, 0.29, 0.43, 0.98)
                # glDrawArrays(GL_TRIANGLES, 0, 36)
        
        # draw animation bvh
        
        # # draw current frame
        # glBindVertexArray(vao_frame)
        # glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

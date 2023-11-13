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

g_cam_location = glm.vec3(4*0.8,4*0.9,4*1.5)
g_cam_location_norm = glm.normalize(g_cam_location)

azimuth = np.arctan(g_cam_location_norm.z/g_cam_location_norm.x)
elevation = np.arcsin(g_cam_location_norm.y)

g_cam_w = glm.vec3(0,0,0)
g_cam_u = glm.vec3(0,0,0)
g_cam_v = glm.vec3(0,0,0)
g_cam_up = glm.vec3(0,1,0)

size = 30
size_t = size
firstMouse = 0
lastX = 0
lastY = 0

single_mesh = 0
single_mesh_index = 0

path = None
path_before = None

toggle_solid_mode = 0

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
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 0.7);

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
    vec3 light_pos = vec3(3,4,5);
    vec3 light_color = vec3(1,1,1);
    //vec3 material_color = vec3(1,0,0);
    float material_shininess = 32.0;

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
    FragColor = vec4(color, 1.);
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
    global mode, animation_mode, single_mesh, toggle_solid_mode
    if key==GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                mode = (mode+1)%2
            if key==GLFW_KEY_H:
                single_mesh = 2
            if key==GLFW_KEY_Z:
                toggle_solid_mode = (toggle_solid_mode + 1)%2

def camera_far(p1,p2):
    return np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

def mouse_zoom_callback(window, xoffset, yoffset):
    global g_cam_location, g_cam_w, mode
    if(mode == 0):
        senitive = 0.2
        if((camera_far(g_cam_center, g_cam_location) > 0.15 ) or yoffset < 0):
            g_cam_location = g_cam_location - g_cam_w*yoffset*senitive
    else:
        senitive = 0.2
        if((camera_far(g_cam_center, g_cam_location) > 0.15 ) or yoffset < 0):
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

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  1 , -1 ,  0, 0,-1, # v5
         1 , -1 , -1 ,  0, 0,-1, # v6

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 , -1 , -1 ,  0, 0,-1, # v6
        -1 , -1 , -1 ,  0, 0,-1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 , -1 ,  1 ,  0,-1, 0, # v3
         1 , -1 , -1 ,  0,-1, 0, # v6
         1 , -1 ,  1 ,  0,-1, 0, # v2

        -1 , -1 ,  1 ,  0,-1, 0, # v3
        -1 , -1 , -1 ,  0,-1, 0, # v7
         1 , -1 , -1 ,  0,-1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 ,  1 ,  1, 0, 0, # v2
         1 , -1 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 , -1 , -1 , -1, 0, 0, # v7
        -1 , -1 ,  1 , -1, 0, 0, # v3

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  1 , -1 , -1, 0, 0, # v4
        -1 , -1 , -1 , -1, 0, 0, # v7
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
    vertices = glm.array(glm.float32,
        # position        # color
         -size_t, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         size_t, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, size_t, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -size_t,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, size_t,  0.0, 0.0, 1.0, # z-axis end 
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

def draw_grid(vao_x,vao_z, MVP, MVP_loc):
    glBindVertexArray(vao_x)
    for i in range(1,size*3):
        i -= size*1.5
        if(i != 0):
            MVP_grid_x = MVP * glm.translate(glm.vec3(0, 0, i))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_x))
            glDrawArrays(GL_LINES, 0, 2)

    glBindVertexArray(vao_z)
    for j in range(1,size*3):
        j -= size*1.5
        if(j != 0):
            MVP_grid_z = MVP * glm.translate(glm.vec3(j, 0, 0))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid_z))
            glDrawArrays(GL_LINES, 0, 2)

def framebuffer_size_callback(window, x, y):
    glViewport(0, 0, x, (int)(x*height/width))

def drop_callback(window, paths):
    global single_mesh, single_mesh_index, s_vertices, path, path_before, vao_single_mesh
    path = paths
    vertex_n = []
    vertex = []
    v_vn = []
    face_number = 0
    face_number_3 = 0
    face_number_4 = 0
    face_number_5 = 0
    
    with open(paths[0], 'r') as file:
        file_name = str(paths[0]).split('\\')[-1]
        #file_name = os.path.basename(paths[0])
        print("file name :",file_name)
        single_mesh = 1
        single_mesh_index = 0
        s_vertices = None
        lines = file.readline()
        while lines:
            #print(lines)
            elements = lines.split()
            if len(elements) == 0:
                lines = file.readline()
                continue
            elif elements[0] == "vn":
                single_vn = [float(elements[1]), float(elements[2]), float(elements[3])]
                vertex_n.append(single_vn)
                #print("vn",lines)
            elif elements[0] == "v":
                single_v = [float(elements[1]), float(elements[2]), float(elements[3])]
                vertex.append(single_v)
                #print("v",elements[1], elements[2], elements[3])
            elif elements[0] == "f":
                face_number += 1
                if(len(elements) == 4):
                    face_number_3 += 1
                elif(len(elements) == 5):
                    face_number_4 += 1
                elif(len(elements) >= 6):
                    face_number_5 += 1
                first_ele = elements[1].split('/')
                for i in range(2, len(elements)-1):
                    face_elements1 = elements[i].split('/')
                    face_elements2 = elements[i+1].split('/')
                    v_vn.extend(vertex[int(first_ele[0])-1])
                    v_vn.extend(vertex_n[int(first_ele[2])-1])
                    v_vn.extend(vertex[int(face_elements1[0])-1])
                    v_vn.extend(vertex_n[int(face_elements1[2])-1])
                    v_vn.extend(vertex[int(face_elements2[0])-1])
                    v_vn.extend(vertex_n[int(face_elements2[2])-1])
                    single_mesh_index += 3
                
            lines = file.readline()
        print("Total number of faces :",face_number)
        print("Number of faces with 3 vertices :", face_number_3)
        print("Number of faces with 4 vertices :", face_number_4)
        print("Number of faces with more than 4 vertices :", face_number_5)
        print("")
    # make vertices
    vertices_n = np.array(v_vn, dtype=np.float32)
    s_vertices = glm.array(vertices_n)
    vao_single_mesh = prepare_single_mesh()
    #print("result",s_vertices)
   
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

def read_file(paths):
    vertex_n = []
    vertex = []
    v_vn = []
    with open(paths, 'r') as file:
        #file_name = os.path.basename(paths[0])
        mesh_index = 0
        lines = file.readline()
        while lines:
            #print(lines)
            elements = lines.split()
            if len(elements) == 0:
                lines = file.readline()
                continue
            elif elements[0] == "vn":
                single_vn = [float(elements[1]), float(elements[2]), float(elements[3])]
                vertex_n.append(single_vn)
                #print("vn",lines)
            elif elements[0] == "v":
                single_v = [float(elements[1]), float(elements[2]), float(elements[3])]
                vertex.append(single_v)
                #print("v",elements[1], elements[2], elements[3])
            elif elements[0] == "f":
                first_ele = elements[1].split('/')
                for i in range(2, len(elements)-1):
                    face_elements1 = elements[i].split('/')
                    face_elements2 = elements[i+1].split('/')
                    v_vn.extend(vertex[int(first_ele[0])-1])
                    v_vn.extend(vertex_n[int(first_ele[2])-1])
                    v_vn.extend(vertex[int(face_elements1[0])-1])
                    v_vn.extend(vertex_n[int(face_elements1[2])-1])
                    v_vn.extend(vertex[int(face_elements2[0])-1])
                    v_vn.extend(vertex_n[int(face_elements2[2])-1])
                    mesh_index += 3
                
            lines = file.readline()
    # make vertices
    vertices_n = np.array(v_vn, dtype=np.float32)
    file_vertices = glm.array(vertices_n)
    return (file_vertices,mesh_index)


def animation_setting():
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    mug_cup_file = "mug_cup.obj"
    table_file = "table.obj"
    cd_file = "cd.obj"

    mug_cup_path = os.path.join(current_dir, mug_cup_file)
    table_path = os.path.join(current_dir, table_file)
    cd_path = os.path.join(current_dir, cd_file)

    (mug_cup_v,mug_index) = read_file(mug_cup_path)
    (table_v,table_index) = read_file(table_path)
    (cd_v,cd_index) = read_file(cd_path)
    return (animation_mesh(mug_cup_v),mug_index,animation_mesh(table_v),table_index,animation_mesh(cd_v),cd_index)



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
    vao_triangle = prepare_vao_triangle()

    # prepare vao for animation

    (vao_mug,mug_index,vao_table,table_index,vao_cd,cd_index) = animation_setting()

    base = Node(None, glm.mat4(), glm.scale((2,-0.5,2)), glm.vec3(0.58,0.29,0))
    arm1 = Node(base, glm.translate(glm.vec3(5.5,0,5.5)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.1,0.1,0.1))
    arm1_leaf1 = Node(arm1, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.5,0.8,0))
    arm1_leaf2 = Node(arm1, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.4,0.2,0))
    arm1_leaf3 = Node(arm1, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.5,0.4,0))
    arm1_leaf4 = Node(arm1, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.2,.1,0))
    
    arm2 = Node(base, glm.translate(glm.vec3(-5.5,0,5.5)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.2,0.2,0.2))
    arm2_leaf1 = Node(arm2, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.2,0.4,0.2))
    arm2_leaf2 = Node(arm2, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.3,0.2,0.5))
    arm2_leaf3 = Node(arm2, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.8,0.5,0.1))
    arm2_leaf4 = Node(arm2, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.1,0.1,0.6))

    arm3 = Node(base, glm.translate(glm.vec3(5.5,0,-5.5)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.3,0.3,0.3))
    arm3_leaf1 = Node(arm3, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.7,0.1,0.2))
    arm3_leaf2 = Node(arm3, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.3,0.2,0.2))
    arm3_leaf3 = Node(arm3, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.2,0.8,0.3))
    arm3_leaf4 = Node(arm3, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.9,0.3,0.1))


    arm4 = Node(base, glm.translate(glm.vec3(-5.5,0,-5.5)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.4,0.4,0.4))
    arm4_leaf1 = Node(arm4, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.9,0.6,0.7))
    arm4_leaf2 = Node(arm4, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.7,0.9,0.8))
    arm4_leaf3 = Node(arm4, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.4,0.8,0.8))
    arm4_leaf4 = Node(arm4, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.3,0.9,0.6))

    arm5 = Node(base, glm.translate(glm.vec3(8,0,0)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.5,0.5,0.5))
    arm5_leaf1 = Node(arm5, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.4,0.5,0.7))
    arm5_leaf2 = Node(arm5, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.5,0.1,0.2))
    arm5_leaf3 = Node(arm5, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.3,0.8,0.1))
    arm5_leaf4 = Node(arm5, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.6,0.3,0.4))

    arm6 = Node(base, glm.translate(glm.vec3(0,0,8)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.6,0.6,0.6))
    arm6_leaf1 = Node(arm6, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.5,0.5,0.5))
    arm6_leaf2 = Node(arm6, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.7,0.2,0.1))
    arm6_leaf3 = Node(arm6, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.3,0.2,0.2))
    arm6_leaf4 = Node(arm6, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.1,0.2,0.1))

    arm7 = Node(base, glm.translate(glm.vec3(-8,0,0)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.7,0.7,0.7))
    arm7_leaf1 = Node(arm7, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.8,0.8,0.5))
    arm7_leaf2 = Node(arm7, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.7,0.7,0.9))
    arm7_leaf3 = Node(arm7, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.7,0.6,0.5))
    arm7_leaf4 = Node(arm7, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.8,0.5,0.8))

    arm8 = Node(base, glm.translate(glm.vec3(0,0,-8)), glm.translate((0,0.033,0))*glm.scale((2.5,1,2.5)), glm.vec3(0.8,0.8,0.8))
    arm8_leaf1 = Node(arm8, glm.translate(glm.vec3(1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.4,0.4,0.2))
    arm8_leaf2 = Node(arm8, glm.translate(glm.vec3(1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.5,0.9,0.1))
    arm8_leaf3 = Node(arm8, glm.translate(glm.vec3(-1,0.,1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.6,0.6,0.2))
    arm8_leaf4 = Node(arm8, glm.translate(glm.vec3(-1,0.,-1)), glm.translate((0,0.4,0))*glm.scale((0.4,0.4,0.4)), glm.vec3(0.9,0.9,0.4))

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        global g_cam_w,g_cam_u,g_cam_v,g_cam_location,g_cam_center,g_cam_up
        
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

        
        # draw triangle w.r.t. the current frame
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # draw grid
        draw_grid(vao_grid_x,vao_grid_z,MVP,MVP_loc)

        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        if(single_mesh == 0):
            glUseProgram(shader_program_light)
            glBindVertexArray(vao_triangle)
            glUniformMatrix4fv(MVP_loc_light, 1, GL_FALSE, glm.value_ptr(MVP))
            glUniformMatrix4fv(M_loc_light, 1, GL_FALSE, glm.value_ptr(M))
            glUniform3f(view_pos_loc_light, g_cam_location.x, g_cam_location.y, g_cam_location.z)
            glUniform3f(color_loc_light, 0.3, 0.3, 0.3)
            glDrawArrays(GL_TRIANGLES, 0, 36)

        elif(single_mesh == 1):
            glUseProgram(shader_program_light)
            glUniformMatrix4fv(MVP_loc_light, 1, GL_FALSE, glm.value_ptr(MVP))
            glUniformMatrix4fv(M_loc_light, 1, GL_FALSE, glm.value_ptr(M))
            glUniform3f(view_pos_loc_light, g_cam_location.x, g_cam_location.y, g_cam_location.z)
            glUniform3f(color_loc_light, 1.0, 0.0, 0.0)
            glBindVertexArray(vao_single_mesh)
            glDrawArrays(GL_TRIANGLES, 0, single_mesh_index)
        
        elif(single_mesh == 2):
            t = glfwGetTime()

            base.set_joint_transform(glm.rotate(t, (0,1,0))*glm.translate(glm.vec3(2,0,0)))
            
            arm1.set_joint_transform(glm.rotate(2*t, (0,1,0)))
            arm1_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm1_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm1_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm1_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))
            
            arm2.set_joint_transform(glm.rotate(1.7*t, (0,1,0)))
            arm2_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm2_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm2_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm2_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))
            
            arm3.set_joint_transform(glm.rotate(1.5*t, (0,1,0)))
            arm3_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm3_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm3_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm3_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))
            
            arm4.set_joint_transform(glm.rotate(2.1*t, (0,1,0)))
            arm4_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm4_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm4_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm4_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))
            
            arm5.set_joint_transform(glm.rotate(2.2*t, (0,1,0)))
            arm5_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm5_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm5_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm5_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))
            
            arm6.set_joint_transform(glm.rotate(1.8*t, (0,1,0)))
            arm6_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm6_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm6_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm6_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))

            
            arm7.set_joint_transform(glm.rotate(1.9*t, (0,1,0)))
            arm7_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm7_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm7_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm7_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))

            arm8.set_joint_transform(glm.rotate(2*t, (0,1,0)))
            arm8_leaf1.set_joint_transform(glm.rotate(4*t+0.5, (0,1,0)))
            arm8_leaf2.set_joint_transform(glm.rotate(4*t+1, (0,1,0)))
            arm8_leaf3.set_joint_transform(glm.rotate(4*t+1.5, (0,1,0)))
            arm8_leaf4.set_joint_transform(glm.rotate(4*t+2, (0,1,0)))

            base.update_tree_global_transform()

            glUseProgram(shader_program_light)
            # glUniformMatrix4fv(MVP_loc_light, 1, GL_FALSE, glm.value_ptr(MVP))
            # glUniformMatrix4fv(M_loc_light, 1, GL_FALSE, glm.value_ptr(M))
            glUniform3f(view_pos_loc_light, g_cam_location.x, g_cam_location.y, g_cam_location.z)
            draw_node(vao_table,table_index,base,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            #draw_node(vao_cd,cd_index,arm2,P*V,MVP_loc_light,M_loc_light,color_loc_light)

            draw_node(vao_cd,cd_index,arm1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm1_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm1_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm1_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm1_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)

            draw_node(vao_cd,cd_index,arm2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm2_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm2_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm2_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm2_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)

            draw_node(vao_cd,cd_index,arm3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm3_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm3_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm3_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm3_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)

            draw_node(vao_cd,cd_index,arm4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm4_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm4_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm4_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm4_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            
            draw_node(vao_cd,cd_index,arm5,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm5_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm5_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm5_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm5_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            
            draw_node(vao_cd,cd_index,arm6,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm6_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm6_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm6_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm6_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            
            draw_node(vao_cd,cd_index,arm7,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm7_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm7_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm7_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm7_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            
            draw_node(vao_cd,cd_index,arm8,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm8_leaf1,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm8_leaf2,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm8_leaf3,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            draw_node(vao_mug,mug_index,arm8_leaf4,P*V,MVP_loc_light,M_loc_light,color_loc_light)
            
        
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

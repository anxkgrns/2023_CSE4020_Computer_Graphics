from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np


width = 1900
height = 1000
pm = 1

mode = 0

g_cam_center = glm.vec3(0,0,0)

g_cam_location = glm.vec3(2*0.8,2*0.9,2*1.5)
g_cam_location_norm = glm.normalize(g_cam_location)

azimuth = np.arctan(g_cam_location_norm.z/g_cam_location_norm.x)
elevation = np.arcsin(g_cam_location_norm.y)

g_cam_w = glm.vec3(0,0,0)
g_cam_u = glm.vec3(0,0,0)
g_cam_v = glm.vec3(0,0,0)
g_cam_up = glm.vec3(0,1,0)

size = 300
size_t = size/10
firstMouse = 0
lastX = 0
lastY = 0


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
    global mode
    if key==GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                mode = (mode+1)%2

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
        # position        # color
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2,-0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        -0.2, 0.2, 0.2, 0.7, 0.7, 0.7, # v0
        0.2,-0.2, 0.2, 0.7, 0.7, 0.7, # v0
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

def prepare_vao_xy_darkgrid(x,z):
    global size
    temp_rad = np.arccos(x/size_t)
    x_scale = size_t*np.sin(temp_rad)
    
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         x, 0.0, -x_scale,  1.0, 1.0, 1.0, # x-axis start
         x, 0.0, x_scale,  1.0, 1.0, 1.0, # x-axis end 
         -x_scale, 0.0, z,  1.0, 1.0, 1.0, # x-axis start
         x_scale, 0.0, z,  1.0, 1.0, 1.0, # x-axis end 
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

def prepare_vao_xz_grid(x,z):
    global size

    temp_rad = np.arccos(x/size_t)
    x_scale = size_t*np.sin(temp_rad)

    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         x, 0.0, x_scale,  0.5, 0.5, 0.5, # x-axis start
         x, 0.0, -x_scale,  0.5, 0.5, 0.5, # x-axis end 
         x_scale, 0.0, z,  0.5, 0.5, 0.5, # x-axis start
         -x_scale, 0.0, z,  0.5, 0.5, 0.5, # x-axis end 
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

def framebuffer_size_callback(window, x, y):
    glViewport(0, 0, x, (int)(x*height/width))

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
    glfwSetKeyCallback(window, key_callback)
    #glfwSetInputMode(window, GLFW_CURSOR,GLFW_CURSOR_DISABLED)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetCursorPosCallback(window, mouse_callback)
    glfwSetScrollCallback(window, mouse_zoom_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_frame()

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
            P = glm.ortho(-size_t/8*width/height,size_t/8*width/height,-size_t/8,size_t/8,-10000,10000)
        
        # render

        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        

        
        #new_g_cam_u = g_cam_u
        #new_g_cam_u.y = 0
        #new_g_cam_u = glm.normalize(new_g_cam_u)
        #g_cam_up = glm.cross(g_cam_w, new_g_cam_u)



        V = glm.lookAt(g_cam_location, g_cam_center, g_cam_up)

        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw grid
        for i in range(size):
            i -= size/2
            if(i%5 == 0):
                i /= 5
                if(i==0):
                    glBindVertexArray(vao_frame)
                    glDrawArrays(GL_LINES, 0, 6)
                else:
                    vao_x_grid = prepare_vao_xy_darkgrid(i, i)
            else:
                i /= 5
                vao_x_grid = prepare_vao_xz_grid(i, i)             
            glBindVertexArray(vao_x_grid)
            glDrawArrays(GL_LINES, 0, 4)
        

        # draw current frame

        M = I

        # current frame: P*V*M
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw triangle w.r.t. the current frame
        glBindVertexArray(vao_triangle)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

/*  Copyright © 2018, Roboti LLC

    This file is licensed under the MuJoCo Resource License (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.roboti.us/resourcelicense.txt
*/


#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "eigenUtils.h"

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <array>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

bool firstLoop = true;
bool pushNow = true;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
        pushNow = true;
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

// controller callback funtcion
void controllerCallback(const mjModel* m, mjData* d) // + (, mjtNum* jacp, mjtNum* jacr) ?
{
    using namespace Eigen;
    using namespace std;

//--------------------------------------------------------------------------------------------

    //Might be Usefull
//    m->stat.meanmass = m_b; //mean mass of the body
//    m_b2 = mj_getTotalmass(m); // which one is right?

//    m->body_ipos; //local position of center of mass         (nbody x 3)

//    d->crb; // com-based composite inertia and mass     (nbody x 10)
//    d->cvel; // com-based velocity [3D rot; 3D tran]     (nbody x 6)
//    mj_jac(m, d, jacp, jacr, point[3], m->nv); // calculating com jacobian

//---------------------------------------------------------------------------------------------
    //Precalculations

    //  COM Jacobian
            mjtNum mi, m_tot;
            mjtNum *jacp_i, *jacr_i;
            mjtNum *jacp_COM = new mjtNum[3 * (m->nv)]{}, *jacr_COM = new mjtNum[3 * (m->nv)]{};

            for (int i = 1; i< m->nbody;i++){
                m_tot += m->body_mass[i];
            }

            for (int i = 1; i < m->nbody; i++) {    // exclude i=0 the world body
                mi = m->body_mass[i];
                mi = mi / m_tot;
                mj_jacBodyCom(m, d, jacp_i, jacr_i, i);
                mju_addToScl(jacp_COM, jacp_i, mi, 3 * (m->nv)); // What is the equivalent to this using Eigen?
                mju_addToScl(jacr_COM, jacr_i, mi, 3 * (m->nv));
            }
    //---


    //Readout
//    for (int i = 0; i < m->nv; i++){
//    cout << i << ": " << d->qvel[i] << "\t"; //how can we check if the values are correct? Should we load another XML File?
//    }
//    cout << endl << endl;
    //------


    //Mappings
    VectorXd qpos = VectorXd::Zero(m->nq);
    qpos = Map<VectorXd> (d->qpos,m->nq);

    //  Mapping Jacobi_COM
    Matrix<float, Dynamic, Dynamic> Jacp_COM (3, m->nv);
    Map<MatrixXf> Jacp_COM (jacp_COM);

    Matrix<float, Dynamic, Dynamic> Jacr_COM (3, m->nv);
    Map<MatrixXf> Jacr_COM (jacr_COM);

    //--

    //Direct Control of the Actuators for Illustration
//    for(int i=0;i < m->nu; i++) {
//        d->qfrc_applied[i+6]= 100.0*(m->qpos0[i+7] - d->qpos[i+7]) - 1.0*d->qvel[i+6] + d->qfrc_bias[i+6];
//    }

}




// main function
int main(int argc, const char** argv)
{
    // check command-line arguments
    if( argc!=2 )
    {
        printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // activate software
    mj_activate("mjkey.txt");

    // load and compile model
    char error[1000] = "Could not load binary model";
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    //Options for solver
    m->opt.jacobian = mjJAC_DENSE;
    m->opt.cone = mjCONE_ELLIPTIC;

    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);



    // set initial values for the floating base
    m->qpos0[2] = 1.1;

    // set initial values for joints
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_hip_x")]     = 0.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_hip_z")]     = 0.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_hip_y")]     = 65.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_knee")]      = 30.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_ankle_y")]   = 65.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "right_ankle_x")]   = 0.0;

    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_hip_x")]      = 0.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_hip_z")]      = 0.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_hip_y")]      = 65.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_knee")]       = 30.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_ankle_y")]    = 65.0*  M_PI/180.0;
    m->qpos0[6+mj_name2id(m, mjOBJ_JOINT, "left_ankle_x")]    = 0.0;

    mj_resetData(m, d);

    mjcb_control = controllerCallback;


    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
            mj_step(m, d);

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}


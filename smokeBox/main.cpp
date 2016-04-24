#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void initGlut(int argc, char *argv[]);
void launchGenerateCheckerboard(dim3 grid, dim3 block, float3 *pos, float3 *norm, int granularity);

// Callbacks
void display();
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// GL resources
GLuint voxels;
GLuint normals;

// CUDA resources
cudaGraphicsResource *cVoxels;
cudaGraphicsResource *cNormals;
float3 *dVoxels;
float3 *dNormals;

// Variables to keep track of camera
int mouseClick = 0;
int mouseX = 0;
int mouseY = 0;
float rotateX = 0;
float rotateY = 0;
float translateX = 0;
float translateY = 0;
float translateZ = -5;

// Constraints of simulation
int length = 20;
int maxVert = length * length * length * 36;

int main(int argc, char *argv[]) {
    // Initialize GL and glut
    initGlut(argc, argv);

    // Initialize CUDA device with GL interoperability
    cudaGLSetGLDevice(0);

    // Initialize the voxel vertex and normal array
    glGenBuffers(1, &voxels);
    glBindBuffer(GL_ARRAY_BUFFER, voxels);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * maxVert * 3, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &normals);
    glBindBuffer(GL_ARRAY_BUFFER, normals);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * maxVert * 3, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Bind VBOs as CUDA resources
    cudaGraphicsGLRegisterBuffer(&cVoxels, voxels, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cNormals, normals, cudaGraphicsMapFlagsWriteDiscard);

    // Map GPU memory and CUDA resources
    cudaMalloc((void **)&(dVoxels), sizeof(float) * maxVert * 3);
    cudaMalloc((void **)&(dNormals), sizeof(float) * maxVert * 3);
    size_t numBytes;
    cudaGraphicsMapResources(1, &cVoxels, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dVoxels, &numBytes, cVoxels);
    cudaGraphicsMapResources(1, &cNormals, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dNormals, &numBytes, cNormals);

    // Callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);

    // Compute checkerboard pattern into pos and normal arrays
    int gridLen = (length + 8 - 1) / 8;
    dim3 grid(gridLen, gridLen, gridLen);
    dim3 block(8, 8, 8);
    launchGenerateCheckerboard(grid, block, dVoxels, dNormals, length);
    cudaGraphicsUnmapResources(1, &cNormals, 0);
    cudaGraphicsUnmapResources(1, &cVoxels, 0);

    glutMainLoop();
    return 0;
}

void initGlut(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutCreateWindow("CUDA sandbox");

    glewInit();

    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Fixed function lighting
    float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    float diffuse[] = { 0.9f, 0.9f, 0.9f, 1.0f };
    float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translateX, translateY, translateZ);
    glRotatef(rotateX, 1.0, 0.0, 0.0);
    glRotatef(rotateY, 0.0, 1.0, 0.0);
    glPushMatrix();
    glRotatef(180.0, 0.0, 1.0, 0.0);

    // Prepare to render
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindBuffer(GL_ARRAY_BUFFER, voxels);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER_ARB, normals);
    glNormalPointer(GL_FLOAT, sizeof(float) * 3, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    // Render
    glEnable(GL_LIGHTING);
    glDrawArrays(GL_TRIANGLES, 0, maxVert * 3);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glPopMatrix();

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    // Set up projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.001, 1000.0);

    glViewport(0, 0, w, h);
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouseClick |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouseClick = 0;
    }

    mouseX = x;
    mouseY = y;
}

void motion(int x, int y)
{
    float dx = (float)(x - mouseX);
    float dy = (float)(y - mouseY);

    if (mouseClick == 1)
    {
        rotateX += dy * 0.2f;
        rotateY += dx * 0.2f;
    }
    else if (mouseClick == 2)
    {
        translateX += dx * 0.01f;
        translateY -= dy * 0.01f;
    }
    else if (mouseClick == 3)
    {
        translateZ += dy * 0.01f;
    }

    mouseX = x;
    mouseY = y;
    glutPostRedisplay();
}
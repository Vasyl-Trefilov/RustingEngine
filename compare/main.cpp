#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <string>

const int STAR_COUNT = 100000;

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in mat4 instanceMatrix;

uniform mat4 projection;
uniform mat4 view;

void main() {
    gl_Position = projection * view * instanceMatrix * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

struct Star {
    glm::vec3 originalPos;
};

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1900, 1080, "OpenGL Starfield", NULL, NULL);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    float vertices[] = {
         0.0f,  0.1f, 0.0f,
        -0.1f, -0.1f, 0.0f,
         0.1f, -0.1f, 0.0f
    };

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    std::vector<Star> stars(STAR_COUNT);
    for (int i = 0; i < STAR_COUNT; i++) {
        float r = 100.0f;
        float theta = (float)rand() / RAND_MAX * 2.0f * 3.14159f;
        float phi = acos(2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        stars[i].originalPos = glm::vec3(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));
    }

    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, STAR_COUNT * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);

    for (int i = 0; i < 4; i++) {
        glEnableVertexAttribArray(1 + i);
        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
        glVertexAttribDivisor(1 + i, 1);
    }

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 1000.0f);
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -300.0f));
    
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    std::vector<glm::mat4> matrices(STAR_COUNT);

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            std::string title = "OpenGL Starfield - FPS: " + std::to_string(nbFrames);
            glfwSetWindowTitle(window, title.c_str());
            nbFrames = 0;
            lastTime += 1.0;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update matrices
        float time = (float)glfwGetTime();
        for (int i = 0; i < STAR_COUNT; i++) {
            float angle = time * 0.5f;
            glm::vec3 p = stars[i].originalPos;
            float x = p.x * cos(angle) - p.z * sin(angle);
            float z = p.x * sin(angle) + p.z * cos(angle);
            
            matrices[i] = glm::translate(glm::mat4(1.0f), glm::vec3(x, p.y, z));
        }

        // Upload and Draw
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, STAR_COUNT * sizeof(glm::mat4), &matrices[0]);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glBindVertexArray(VAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, STAR_COUNT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
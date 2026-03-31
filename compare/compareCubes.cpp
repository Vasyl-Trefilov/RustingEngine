#include "raylib.h"
#include "raymath.h"
#include <vector>
#include <cmath>

#define GRID_SIZE 20
#define HEIGHT_SIZE 25
#define TOTAL_CUBES (GRID_SIZE * HEIGHT_SIZE * GRID_SIZE)

struct CubeData {
    Vector3 pos;
    Vector3 vel;
};

int main() {
    InitWindow(1280, 720, "Raylib - CPU Physics (10k Objects)");
    SetTargetFPS(60);

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 30.0f, -50.0f, 30.0f };
    camera.target = (Vector3){ 0.0f, -5.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // 2. Setup Meshes and Materials
    Mesh cubeMesh = GenMeshCube(1.0f, 1.0f, 1.0f);
    
    // Raylib instancing only allows 1 color per batch.
    // To replicate your alternating colors, we need TWO batches.
    Material redMat = LoadMaterialDefault();
    redMat.maps[MATERIAL_MAP_ALBEDO].color = (Color){ 255, 51, 51, 255 }; // 1.0, 0.2, 0.2
    
    Material greenMat = LoadMaterialDefault();
    greenMat.maps[MATERIAL_MAP_ALBEDO].color = (Color){ 51, 255, 51, 255 }; // 0.2, 1.0, 0.2

    std::vector<CubeData> redCubes;
    std::vector<CubeData> greenCubes;
    
    // 3. Spawn the 10,000 Cubes
    for (int x = 0; x < GRID_SIZE; x++) {
        for (int y = 0; y < HEIGHT_SIZE; y++) {
            for (int z = 0; z < GRID_SIZE; z++) {
                CubeData c;
                c.pos.x = (x - GRID_SIZE / 2.0f) * 1.5f;
                c.pos.y = y * 1.5f + 10.0f;
                c.pos.z = (z - GRID_SIZE / 2.0f) * 1.5f;
                c.vel = {0.0f, 0.0f, 0.0f};

                if ((x + y + z) % 2 == 0) {
                    redCubes.push_back(c);
                } else {
                    greenCubes.push_back(c);
                }
            }
        }
    }

    // Allocate GPU transform matrices
    Matrix* redTransforms = (Matrix*)MemAlloc(redCubes.size() * sizeof(Matrix));
    Matrix* greenTransforms = (Matrix*)MemAlloc(greenCubes.size() * sizeof(Matrix));

    // 4. MAIN LOOP
    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (dt > 0.033f) dt = 0.033f; // Cap delta time to prevent physics explosions if it lags

        // --- CPU PHYSICS PASS ---
        // We combine the arrays via pointers just to loop over all of them easily
        std::vector<CubeData*> allCubes;
        for (auto& c : redCubes) allCubes.push_back(&c);
        for (auto& c : greenCubes) allCubes.push_back(&c);

        for (size_t i = 0; i < allCubes.size(); i++) {
            CubeData* me = allCubes[i];

            // Gravity Integration
            me->vel.y -= 9.81f * dt;
            me->pos.x += me->vel.x * dt;
            me->pos.y += me->vel.y * dt;
            me->pos.z += me->vel.z * dt;

            // Collision against Floor (Y = 3.5 is the top of the 40x1x40 floor at Y=3.0)
            if (me->pos.y < 3.5f) {
                me->pos.y = 3.5f;
                if (me->vel.y < 0.0f) {
                    me->vel.y *= -0.2f; // Bounce
                    me->vel.x *= 0.8f;  // Friction
                    me->vel.z *= 0.8f;
                }
            }

            // NAIVE O(N^2) CPU BROADPHASE (This is what makes C++ CPU physics lag!)
            for (size_t j = i + 1; j < allCubes.size(); j++) {
                CubeData* other = allCubes[j];

                float dx = other->pos.x - me->pos.x;
                float dy = other->pos.y - me->pos.y;
                float dz = other->pos.z - me->pos.z;
                float distSq = (dx*dx) + (dy*dy) + (dz*dz);

                // Simple sphere overlap (radius = 0.5 + 0.5 = 1.0)
                if (distSq < 1.0f && distSq > 0.0001f) {
                    float dist = sqrt(distSq);
                    float overlap = 1.0f - dist;
                    
                    float nx = dx / dist;
                    float ny = dy / dist;
                    float nz = dz / dist;

                    // Push objects apart
                    me->pos.x -= nx * overlap * 0.5f;
                    me->pos.y -= ny * overlap * 0.5f;
                    me->pos.z -= nz * overlap * 0.5f;

                    other->pos.x += nx * overlap * 0.5f;
                    other->pos.y += ny * overlap * 0.5f;
                    other->pos.z += nz * overlap * 0.5f;

                    // Simple velocity dampening on hit
                    me->vel.x *= 0.9f; me->vel.z *= 0.9f;
                    other->vel.x *= 0.9f; other->vel.z *= 0.9f;
                }
            }
        }

        // --- UPDATE MATRICES FOR GPU INSTANCING ---
        for (size_t i = 0; i < redCubes.size(); i++) {
            redTransforms[i] = MatrixTranslate(redCubes[i].pos.x, redCubes[i].pos.y, redCubes[i].pos.z);
        }
        for (size_t i = 0; i < greenCubes.size(); i++) {
            greenTransforms[i] = MatrixTranslate(greenCubes[i].pos.x, greenCubes[i].pos.y, greenCubes[i].pos.z);
        }

        // --- RENDER PASS ---
        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(camera);

        // Draw the massive floor
        DrawCube((Vector3){0.0f, 3.0f, 0.0f}, 40.0f, 1.0f, 40.0f, GRAY);
        DrawCubeWires((Vector3){0.0f, 3.0f, 0.0f}, 40.0f, 1.0f, 40.0f, DARKGRAY);

        // Draw the 10,000 cubes using GPU Instancing
        DrawMeshInstanced(cubeMesh, redMat, redTransforms, redCubes.size());
        DrawMeshInstanced(cubeMesh, greenMat, greenTransforms, greenCubes.size());

        EndMode3D();
        DrawFPS(10, 10);
        DrawText("CPU Physics Running...", 10, 40, 20, DARKGRAY);
        EndDrawing();
    }

    // 5. Cleanup
    MemFree(redTransforms);
    MemFree(greenTransforms);
    UnloadMaterial(redMat);
    UnloadMaterial(greenMat);
    UnloadMesh(cubeMesh);
    CloseWindow();

    return 0;
}
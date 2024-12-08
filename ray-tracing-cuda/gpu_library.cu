#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string.h>

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "hitable_list.h"

namespace py = pybind11;


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable **world, int max_depth, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < max_depth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(2024+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, int max_depth, curandState *rand_state) {
    
    // std::cout<<ns<<std::endl;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, max_depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(float *buffer, hitable **d_list, hitable **d_world,
                             int width, int height, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        buffer[0] = 2024;
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);
    }
}

__global__ void create_camera(float lookfrom_0, float lookfrom_1, float lookfrom_2,
                              float lookat_0, float lookat_1, float lookat_2,
                              float up_0, float up_1, float up_2,
                              float aperture, int width, int height,
                              camera **d_camera){
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        vec3 lookfrom(lookfrom_0, lookfrom_1, lookfrom_2);
        vec3 lookat(lookat_0, lookat_1, lookat_2);
        vec3 up(up_0, up_1, up_2);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        // float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 up,
                                 30.0,
                                 float(width)/float(height),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0){
        for(int i=0; i < 22*22+1+3; i++) {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
        delete *d_world;
    }

}

__global__ void free_camera(camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0){
        delete *d_camera;
    }
}


float render_cuda(py::array_t<float> vec, const py::dict& config)
{

    float *buffer;
    checkCudaErrors(cudaMallocManaged((void **)&buffer, 10));

    py::buffer_info ha = vec.request();
    float* ptr = reinterpret_cast<float*>(ha.ptr);

    int width = config["width"].cast<int>();
    int height = config["height"].cast<int>();
    int ns = config["samples_per_pixel"].cast<int>();
    int max_depth = config["max_depth"].cast<int>();
    float aperture = config["aperture"].cast<float>();

    int tx = 8;
    int ty = 8;

    int num_pixels = width*height;
    size_t fb_size = num_pixels*sizeof(vec3);

    py::array_t<float> lookfrom_arr = py::cast<py::array_t<float>>(config["lookfrom"]);
    float* lookfrom_ptr = static_cast<float*>(lookfrom_arr.request().ptr);

    py::array_t<float> lookat_arr = py::cast<py::array_t<float>>(config["lookat"]);
    float* lookat_ptr = static_cast<float*>(lookat_arr.request().ptr);

    py::array_t<float> up_arr = py::cast<py::array_t<float>>(config["up"]);
    float* up_ptr = static_cast<float*>(up_arr.request().ptr);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(buffer, d_list,d_world, width, height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_camera<<<1,1>>>(lookfrom_ptr[0], lookfrom_ptr[1],lookfrom_ptr[2],
                           lookat_ptr[0], lookat_ptr[1], lookat_ptr[2],
                           up_ptr[0], up_ptr[1], up_ptr[2],
                           aperture, width, height, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(width/tx+1,height/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, width, height, ns, d_camera, d_world, max_depth, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    float duration = ((float)(stop - start)) / CLOCKS_PER_SEC;

    // Output FB as Image
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j*width + i;

            ptr[pixel_index*3] = fb[pixel_index].r();
            ptr[pixel_index*3+1] = fb[pixel_index].g();
            ptr[pixel_index*3+2] = fb[pixel_index].b();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world);
    free_camera<<<1,1>>>(d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));


    cudaDeviceReset();

    return duration;
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("render", render_cuda);
}
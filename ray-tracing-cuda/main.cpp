#include <iostream>
#include <time.h>

void render(float *fb, int max_x, int max_y){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main(){
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx <<"x" << ny <<" image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);


    clock_t start, stop;
    start = clock();

    float *fb;
    render(fb, nx, ny)
    
    stop = clock();
    double timer_seconds = ((double)(stop-start))/CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    std::cout<<"P3\n" << nx <<" " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--){
        for (int i = 0; i < nx; i++){
            size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index+0];
            float g = fb[pixel_index+1];
            float b = fb[pixel_index+2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir <<" "<<ig<< " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(fb));

}
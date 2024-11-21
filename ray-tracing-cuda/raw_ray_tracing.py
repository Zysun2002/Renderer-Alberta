import numpy as np
import ipdb
from tqdm import tqdm
from PIL import Image

class Ray():
    def __init__(self, origin, direction):
        self.orig = origin
        self.dir = direction

    def at(self, t):
        return self.orig + t * self.dir

class Hit_Record:
    def __init__(self):
        self.p, self.normal, self.t = None, None, None
        self.mat = None
        self.front_face = None

    def set_face_normal(self, ray, outward_normal):
        self.front_face = np.dot(ray.dir, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

class Hittable:
    pass
    
class Sphere(Hittable):
    def __init__(self, center, radius, mat):
        super().__init__()
        self.center = center
        self.radius = radius
        self.mat = mat

    def hit(self, ray, ray_t, rec):
        oc = self.center - ray.orig
        a = np.dot(ray.dir, ray.dir)
        h = np.dot(ray.dir, oc)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = h*h - a*c
        
        if discriminant < 0:
            return False
        
        sqrtd = np.sqrt(discriminant)

        root = (h - sqrtd) / a
        if (not ray_t.surrounds(root)):
            root = (h + sqrtd) / a
            if (not ray_t.surrounds(root)):
                return False
            
        rec.t = root
        rec.p = ray.at(rec.t)   
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.mat = self.mat
        return True

        

class Hittable_List(Hittable):
    def __init__(self):
        self.objects = []
    
    def add(self, object):
        self.objects.append(object)
    
    def hit(self, ray, ray_t, rec):
        temp_rec = Hit_Record()
        hit_anything = False
        closest_so_far = ray_t.max

        for object in self.objects:
            if object.hit(ray, Interval(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec          

        return hit_anything, rec
    
class Interval:
    def __init__(self, min, max):
        self.min, self.max = min, max

    def size(self):
        return self.max - self.min
    
    def contains(self, x):
        return self.min <= x and x <= self.max 

    def surrounds(self, x):
        return self.min < x and x < self.max
    
    def clamp(self, x):
        if x < self.min: return self.min
        if x > self.max: return self.max
        return x

class Camera:
    def __init__(self):
        self.aspect_ratio = 1.0
        self.width = 100
        self.samples_per_pixel = 10
        self.max_depth = 50
        self.image, self.pixel00_loc, self.pixel_delta_u, self.pixel_delta_v = \
            None, None, None, None
        
        self.vfov = 90
        self.lookfrom = np.array((0, 0, 0))
        self.lookat = np.array((0, 0, -1))
        self.vup = np.array((0, 1, 0))

        self.defocus_angle = 0
        self.focus_dist = 10

    def render(self, world):
        self.initialize()
        for j in tqdm(range(0, self.height)):
            for i in range(0, self.width):
                pixel_color = np.array((0., 0., 0.))
                for sample in range(self.samples_per_pixel):
                    r = self.get_ray(i, j)
                    pixel_color += self.ray_color(r, self.max_depth, world)
        
                self.write_color(j, i, self.pixel_samples_scale*pixel_color)
            self.image.save("real-time.png")   
                
        self.image.save("output5.png")   

    def initialize(self):
        self.height = int(self.width / self.aspect_ratio)
        self.image = Image.new("RGB", (self.width, self.height), (0, 0, 0))

        self.pixel_samples_scale = 1./self.samples_per_pixel
        
        self.camera_center = self.lookfrom

        # camera
        theta = np.deg2rad(self.vfov)
        h = np.tan(theta/2)
        viewport_height = 2*h*self.focus_dist
        viewport_width = viewport_height * float(self.width)/self.height

        self.w = norm(self.lookfrom - self.lookat)
        self.u = norm(np.cross(self.vup, self.w))
        self.v = np.cross(self.w, self.u)

        viewport_u = viewport_width * self.u
        viewport_v = viewport_height * -self.v

        self.pixel_delta_u = viewport_u / self.width
        self.pixel_delta_v = viewport_v / self.height 

        viewport_upper_left = self.camera_center - self.focus_dist* self.w - viewport_u/2 - viewport_v/2 
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
        
        defocus_radius = self.focus_dist * np.tan(np.deg2rad(self.defocus_angle/2))
        self.defocus_disk_u = self.u * defocus_radius
        self.defocus_disk_v = self.v * defocus_radius

    def get_ray(self, i, j):
        offset = self.sample_square()
        pixel_sample = self.pixel00_loc + \
            (i+offset[0])*self.pixel_delta_u + (j+offset[1])*self.pixel_delta_v
        
        ray_origin = self.camera_center if self.defocus_angle<=0 else self.defocus_disk_sample()
        ray_direction = pixel_sample - ray_origin
        return Ray(ray_origin, ray_direction)

    def defocus_disk_sample(self):
        p = random_in_unit_disk()
        return self.camera_center + p[0] * self.defocus_disk_u + p[1] * self.defocus_disk_v

    def sample_square(self):
        a, b = np.random.uniform(-.5, .5), np.random.uniform(-.5, .5)
        return np.array((a, b, 0))

    def ray_color(self, ray, depth, world):
        if depth <= 0:
            return np.array((0, 0, 0))
        
        rec = Hit_Record()
        hit_anything, rec = world.hit(ray, Interval(0.001, np.inf), rec)
        if (hit_anything):
            is_scatter, scattered, attenuation = rec.mat.scatter(ray, rec)
            if is_scatter:
                return attenuation * self.ray_color(scattered, depth-1, world)
            
            return np.array((0, 0, 0))

        ray = norm(ray.dir) 
        a = 0.5 * (ray[1] + 1.) 
        color = (1-a) * np.array((1., 1., 1.)) + a * np.array((0.5, 0.7, 1.))
        return color 
    
    def write_color(self, j, i, color):
        intensity = Interval(0, 0.999)
        color = tuple(int(intensity.clamp(c) * 256) for c in color)
        self.image.putpixel((i, j), color)


        
class Material:
    def __init__(self):
        pass

    def scatter(self):
        return False
    
class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, _, rec):
        scatter_direction = rec.normal + random_unit_vector()
        if np.all(scatter_direction<1e-8):
            scatter_direction = rec.normal
        return True, Ray(rec.p, scatter_direction), self.albedo

class Metal(Material):
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = fuzz if fuzz < 1 else 1
    
    def scatter(self, r_in, rec):
        reflected = reflect(r_in.dir, rec.normal)
        reflected = norm(reflected) + self.fuzz*random_unit_vector()
        scattered = Ray(rec.p, reflected)
        is_scattered = np.dot(scattered.dir, rec.normal) > 0
        return is_scattered, scattered, self.albedo 

class Dielectric(Material):
    def __init__(self, refraction_index):
        self.refraction_index = refraction_index

    def scatter(self, r_in, rec):
        attenuation = np.array((1., 1., 1.))
        ri = 1./self.refraction_index if rec.front_face else self.refraction_index
        unit_direction = norm(r_in.dir)
        cos_theta = np.minimum(np.dot(-unit_direction, rec.normal), 1.)
        sin_theta = np.sqrt(1. - cos_theta*cos_theta)

        cannot_refract = ri * sin_theta > 1.
        if cannot_refract or self.reflectance(cos_theta, ri) > np.random.uniform(0, 1):
            direction = reflect(unit_direction, rec.normal)
        else: direction = refract(unit_direction, rec.normal, ri)

        scattered = Ray(rec.p, direction)
        return True, scattered, attenuation
    
    def reflectance(self, cosine, refraction_index):
        r0 = (1 - refraction_index) / (1 + refraction_index)
        r0 = r0 * r0
        return r0 + (1-r0) * np.power((1-cosine), 5)

def reflect(v, n):
    return v - 2*np.dot(v,n)*n

def refract(uv, n, etai_over_etat):
    
    cos_theta = np.minimum(np.dot(-uv, n), 1.)
    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -np.sqrt(np.abs(1.-np.dot(r_out_perp, r_out_perp))) * n
    # ipdb.set_trace()
    return r_out_perp + r_out_parallel

def random_unit_vector():
    while(True):
        p = np.random.uniform(-1, 1, size=3)
        lensq = np.linalg.norm(p)
        if 1e-30 < lensq and lensq <= 1:
            return norm(p)

def random_on_hemisphere(normal):
    on_unit_sphere = random_unit_vector()
    if (np.dot(on_unit_sphere, normal) > 0.):
        return on_unit_sphere
    else:
        return -on_unit_sphere   


def random_in_unit_disk():
    while(True):
        p = np.array((np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0))
        if np.dot(p, p) < 1:
            return p

def norm(vec):
    return vec / np.linalg.norm(vec)

def main():

    # world
    world = Hittable_List()

    ground_material = Lambertian(np.array((0.5, 0.5, 0.5)))
    world.add(Sphere(np.array((0, -1000, 0)), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = np.random.uniform(0, 1)
            center = np.array((a+0.9*np.random.uniform(0,1), 0.2, b+0.9*np.random.uniform(0,1)))

            if np.linalg.norm(center - np.array((4, 0.2, 0))) > 0.9:
                sphere_material = Material()

                if choose_mat < 0.8:
                    albedo = np.random.uniform(0, 1, size=3) * np.random.uniform(0, 1, size=3)
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    albedo = np.random.uniform(0, 0.5, size=3)
                    fuzz = np.random.uniform(0, 0.5)
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                else:
                    sphere_material = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material))
        
        material1 = Dielectric(1.5)
        world.add(Sphere(np.array((0, 1, 0)), 1., material1))

        material2 = Lambertian(np.array((0.4, 0.2, 0.1)))
        world.add(Sphere(np.array((-4, 1, 0)), 1., material2))
        
        material3 = Metal(np.array((0.7, 0.6, 0.5)), 0.)
        world.add(Sphere(np.array((4, 1, 0)), 1., material3))
        

    # camera
    camera = Camera()
    camera.aspect_ratio = 16. / 9.
    camera.width = 800
    camera.samples_per_pixel = 100
    camera.max_depth = 50
    
    camera.vfov = 90
    camera.lookfrom = np.array((13, 2, 3))
    camera.lookat = np.array((0, 0, 0))
    camera.vup = np.array((0, 1, 0))

    camera.defocus_angle = 0.6
    camera.focus_dist = 10.
    # render

    camera.render(world)

     

if __name__ == '__main__':
    main()

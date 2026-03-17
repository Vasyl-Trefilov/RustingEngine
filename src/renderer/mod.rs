pub mod camera;
pub mod pipeline;
pub mod swapchain;

use std::sync::Arc;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub struct VulkanBase {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub window: Arc<Window>,
    pub instance: Arc<Instance>,
}

pub fn init_vulkan(event_loop: &EventLoop<()>) -> VulkanBase {
    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    let required_extensions = vulkano_win::required_extensions(&library);

    let instance = Instance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();

    let surface = WindowBuilder::new()
        .with_title("Vulkan Modular Engine")
        .build_vk_surface(event_loop, instance.clone())
        .unwrap();

    let window = surface.object().unwrap().clone().downcast::<Window>().unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                q.queue_flags.intersects(QueueFlags::GRAPHICS) && 
                p.surface_support(i as u32, &surface).unwrap_or(false)
            }).map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            _ => 2,
        }).unwrap();

    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
        ..Default::default()
    }).unwrap();

    let queue = queues.next().unwrap();

    VulkanBase {
        device,
        queue,
        surface,
        window,
        instance,
    }
}
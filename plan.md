Claude, its 2025.

We need to do nvstreamux -> nvinfer batch -> nvosd batch -> dmux -> [rtspout0,rtspout1] 

Deepstream has come out with psyservicemaker

I want to try it. I hate it. but I will learn.

From what i can tell, it has everything you need in the ds8 image

root@4b8d540e79a1:/opt/nvidia/deepstream/deepstream-8.0/service-maker/python# ls
pyservicemaker-0.0.1-cp312-cp312-linux_x86_64.whl

So the plan should be to use the FROM ds8 + add ur code to the cwd of the examples as thats where the build entry point is setup for.

so just mount into the that location and try to get this running

As a first step lets research the right dockerfile and get a helloworld from psyservicemaker

https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_quick_start.html

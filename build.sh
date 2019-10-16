image=kuberlab/tfpose:1.0.0
image_gpu=kuberlab/tfpose:1.0.0-gpu

docker build --tag $image -f Dockerfile.kibernetika .
docker build --tag $image_gpu -f Dockerfile.kubernetika-gpu .

if [ "$1" == "--push" ];
then
    docker push $image
    docker push $image_gpu
fi


image=kuberlab/tfpose:latest
image_gpu=kuberlab/tfpose:latest-gpu

docker build --tag $image -f Dockerfile.kibernetika .
docker build --tag $image_gpu -f Dockerfile.kibernetika-gpu .

if [ "$1" == "--push" ];
then
    docker push $image
    docker push $image_gpu
fi


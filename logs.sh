# get id of running container with image name
# container_id=$(sudo docker ps | grep youtube-transcriber_app_1 | awk '{print $1}')
# echo "container id: $container_id"
# get logs
# sudo docker logs -f $container_id
# connect to container
# sudo docker exec -it $container_id /bin/bash
sudo docker logs -f -t bot_youtube_tts
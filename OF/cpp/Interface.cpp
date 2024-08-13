#include"Interface.h"
#include<sys/socket.h>
#include<string>
#include<vector>
#include<iostream>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<cstring>

using namespace std;

Interface::Interface(int port, char id_)
{
	memset(&addr_send, 0, sizeof(addr_send));
	addr_send.sin_family = AF_INET;
	inet_pton(AF_INET, "192.168.2.2", &(addr_send.sin_addr));
	addr_send.sin_port = htons(port);

	this->id_ = id_;
}


void Interface::send(vector<float> velocity, int quality){
	 
	message.id = id_;
	message.V1=velocity[0];
	message.V2=velocity[1];
	message.V3=velocity[2];
	message.quality = quality;

	int ret = sendto(socket_fd, &message, sizeof(message), 0, (const struct sockaddr*)&addr_send, sizeof(addr_send));

	//if (ret) {
	//	cout<<sizeof(message)<<endl;
	//	cout<<ret<<endl;
	//}
}

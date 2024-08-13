#pragma once
#include<sys/socket.h>
#include<vector>
#include<netinet/in.h>
#include<arpa/inet.h>

class Interface {
	public:
	Interface(int port, char id_);
	void send(std::vector<float> velocity, int quality);	

	private:
	int socket_fd = socket(PF_INET, SOCK_DGRAM, 0);
	char id_;
	struct sockaddr_in addr_send = {};
	struct __attribute__((packed)) Message {
		char id;
		float V1;
		float V2;
		float V3;
		int quality;
	} message;
};

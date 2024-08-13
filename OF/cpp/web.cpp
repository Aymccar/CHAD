#include"OpticalFlowProcessor.h"
#include"cpp-httplib/httplib.h"
#include<fstream>
#include<string>
#include"nlohmann/json.hpp"

#include<iostream>

using namespace httplib;
using namespace std;
using namespace nlohmann;

void OpticalFlowProcessor::new_frame_req(){


	svr.Get("/new_ref", [&](const Request& req, Response& res) {
			string content;
			openFile("www/new_ref/main.html", content);
			res.set_content(content, "text/html");
			});

	svr.Post("/new_ref", [&](const Request& req, Response& res) {
			set_ref();
			return true;
			});
}



void OpticalFlowProcessor::openFile(string path, string& buffer){
	ifstream file(path);

	string temp;
	while(file){
		getline(file, temp);
		buffer += temp;
	}	

}

void OpticalFlowProcessor::config(){


	svr.Get("/config", [&](const Request& req, Response& res) {

			string config;
			paramsToJson(config);

			string content;
			openFile("www/config/main.html", content);
			size_t pos  = content.find("REPLACE_ME");
			content.erase(pos, 10);
			content.insert(pos, config + ";");
			res.set_content(content, "text/html");
			});

	svr.Post("/config", [&](const Request& req, Response& res) {
			string val = req.body;
			auto j = json::parse(val);
			
			params[j["id"]] = stoi((string)j["val"]);
			
			return true;
			});
}


void OpticalFlowProcessor::dashboard(){//TODO with websocket


	svr.Get("/dashboard", [&](const Request& req, Response& res) {
			
			string content;
			openFile("www/dashboard/main.html", content);
			
			size_t pos  = content.find("REPLACE_ME_Vx");
			content.erase(pos, 13);
			content.insert(pos, to_string(Vel[0]));

			pos  = content.find("REPLACE_ME_Vy");
			content.erase(pos, 13);
			content.insert(pos, to_string(Vel[1]));

			pos  = content.find("REPLACE_ME_Vz");
			content.erase(pos, 13);
			content.insert(pos, to_string(Vel[2]));
			
			pos  = content.find("REPLACE_ME_qual");
			content.erase(pos, 15);
			content.insert(pos, to_string(qual));
			
			res.set_content(content, "text/html");
			});

}


void OpticalFlowProcessor::paramsToJson(string& buffer){
	string temp;
	temp += "{";

	map<string, int>::iterator it;
	for(it = params.begin(); it!= params.end(); it++){
		temp +="\"" + it->first + "\": " + "\"" + to_string(it->second) + "\",";
	}
	temp += "}";

	buffer = temp;
}

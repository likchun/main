/**
 * @file IzhikevichModel.cpp
 * @author likchun@outlook.com
 * @brief simulate the dynamics of a network of spiking neurons
 * @version 1.4.6(16)
 * @date 2022-04-27
 * 
 * @copyright
 * 
 * @note to be compiled in C++ version 20 with g++
 * @bug 
 * 
 */

// Features to be implemented:
// forceOverwrite flag


#include <filesystem>
#include <algorithm>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <limits>
#include <chrono>
#include <boost/random.hpp> // Use <boost/random.hpp> instead of <random> for compatibility
#include <vector>
#include <tuple>
#include <cmath>
#include "myinc.h"
#define _CRT_SECURE_NO_WARNINGS

std::string code_ver = "Version 1.4.6 | Build 16 | Last Update 27 Apr 2022";

using namespace std;
using namespace boost;
using namespace myinc;
namespace fs = std::filesystem;
using path = fs::path;

struct Variables
{
	// Modify variables and parameters in 'vars.txt'
	path   infile_vars = "vars.txt";

	// Data Input
	path   infile_path = "";
	path   inadjm_path = "";
	path   infile_adjm = "DIV66_gji.txt";
	string mat_format  = "nonzero";
	int    mat_size    = 4095;
	char   delim       = '\t';

	// Numerical Settings
	double dt          = 0.125;
	double Tn          = 7500;
	float  rand_seed   = 0;
	double trunc_t_inh = 250;
	double trunc_t_exc = 250;

	// Suppression of Synaptic Weights
	double suppr_lv    = 0;
	int    suppr_type  = -1; // +1: suppress exc links, -1: suppress inh links
	vector <int> suppr_nodes;

	// Parameters for Izhikevich's Neuron Model
	double a_inh = 0.10, b_inh = 0.2, c_inh = -65, d_inh = 2;
	double a_exc = 0.02, b_exc = 0.2, c_exc = -65, d_exc = 8;
	double sigma = 3;

	// Parameters for Synapse Model
	double thres_v_inh = -80.0;
	double thres_v_exc = 0.0;
	double tau_inh     = 6;
	double tau_exc     = 5;
	double beta        = 2;

	// Initial Values
	double memp_initval = -65;
	double recv_initval = 8;

	// Data Output
	path outfile_path = "";
	path outfile_spkt = "spkt.txt";
	path outfile_spks = "spks.txt";
	path outfile_info = "info.txt";
	path outfile_cont = "cont.dat";
	bool expoTimeSeri = true;
	path outfile_memp = "memp.dat";
	path outfile_curr = "curr.dat";
	path outfile_recv = "recv.dat";
	path logfile	  = "";
	bool forceOverwrt = false;
	// Avoid modifying these file names

	// Other Settings
	size_t	TIMESERIES_BUFF	= 150000000; // for infinity: numeric_limits<int>::max()
	int		PRECISION_DIGIT	= numeric_limits<float>::max_digits10; // use SINGLE floating point precision for time series output
	// int PRECISION_DIGIT = numeric_limits<double>::max_digits10; // use DOUBLE floating point precision for time series output
	double	prec			= pow(10, PRECISION_DIGIT);
} vars;

const int parser(int, char**, const char*, Variables&);
path set_logfile_name(Variables&, string);
void set_io_path(Variables&);
bool check_file_existance(Variables&);
void display_settings(Variables&);
string remove_whitespace(string&);
const int import_vars(Variables&);
const int import_adjm(vector<vector<double>>&, Variables&);
const int estimate_trunc_t(vector<vector<double>>&, Variables&);
int import_prev_vars(vector<double>&, vector<double>&, vector<double>&, Variables&);
const int import_prev_spkt(vector<vector<double>>&, path&);
const int import_prev_spks(vector<vector<int>>&, path&);
const int classify_neuron_class(vector<int>&, vector<vector<double>>&);
const int create_quick_link_ref(vector<vector<double>>&, vector<vector<int>>&, vector<vector<int>>&);
void setup_hash_table_exp(vector<double>&, vector<double>&, int, int);
void suppress_inhibition(vector<vector<double>>&, double);
void suppress_inhibition_of_selected(vector<vector<double>>&, vector<vector<int>>&, Variables&);
void suppress_excitation(vector<vector<double>>&, double);
void suppress_excitation_of_selected(vector<vector<double>>&, vector<vector<int>>&, Variables&);
const int export_info(int, float, Variables&);
const int export_cont(int, vector<double>&, vector<double>&, vector<double>&, Variables&);
const int export_spkt(vector<vector<double>>&, path&);
const int export_spks(vector<vector<int>>&, path&);
const int export_time_series_bin(vector<float>&, path&, int);
const int throw_error(string);
const int throw_error(string, path&);
const int throw_error(string, string);
const int throw_error(string, string*);
const int throw_error(string, int*);


int main(int argc, char** argv)
{
	const char* optstr = "hw:i:o:l:";
	if (parser(argc, argv, optstr, vars) == EXIT_FAILURE) { return EXIT_FAILURE; }


	int mode = 0;			// 0: overwrite mode | 1: continue mode
	int continuation = -1;	// count the number of times of continuation

	mt19937 random_generator;
	normal_distribution <double> norm_dist(0, 1);

	size_t	TIMESERI_RESERVE_SIZE;
	int		now_step, diff_step, total_step;
	int		trunc_step_inh, trunc_step_exc;
	double	noise, spike_sum, memp_temp;
	double	conductance_inh, conductance_exc;

	vector <int>		neuron_type;
	vector <float>		expo_memp;
	vector <double>		membrane_potential, recovery_variable, synaptic_current;
	vector <double>		HASHTABLE_EXP_INH, HASHTABLE_EXP_EXC;

	vector <vector<int>>	inh_links, exc_links;	// `inh_links[i][j]` stores the j-th inhibitory incoming link for node i (i, j starts from 0)
	vector <vector<int>>	spike_timesteps;
	vector <vector<double>>	spike_timestamps, synaptic_weights;


	auto [time_fmt1, time_fmt2] = timer.report_time();
	outlog.open(set_logfile_name(vars, time_fmt2));

	outlog << '\n' << code_ver << "\n\n" << time_fmt1 << '\n';
	outlog << "[Initialization] starts\n";
	timer.stopwatch_begin();	

	set_io_path(vars);
	if (check_file_existance(vars) && !vars.forceOverwrt)
	{
		string user_input;
		outlog << "\n<ALERT> existing data files in targeted directory\n"
			   << "- press ENTER to continue from previous data, or\n"
			   << "- insert 'overwrite' to replace current data files\n\n"
			   << "* note1: directory of synaptic weight file should be\n"
			   << "         specified (by using the flag [-w]) or in the\n"
			   << "         same directory as the program\n"
			   << "* note2: synaptic weight file name should not change\n";
		getline(cin, user_input);
		if (user_input.empty()) { mode = 1; }
		else if (user_input == "overwrite") {
			vars.forceOverwrt = true;
			outlog << '\n';
		}
		else { return throw_error("invalid_input", user_input); }
	}

	if (mode == 0)			// overwrite mode
	{
		if (import_vars(vars) == EXIT_FAILURE) { return EXIT_FAILURE; }
		if (import_adjm(synaptic_weights, vars) == EXIT_FAILURE) { return EXIT_FAILURE; }
		vars.mat_size = synaptic_weights.size();
		membrane_potential = vector<double>(vars.mat_size);
		recovery_variable = vector<double>(vars.mat_size);
		synaptic_current = vector<double>(vars.mat_size);
		spike_timestamps = vector<vector<double>>(vars.mat_size);
		spike_timesteps = vector<vector<int>>(vars.mat_size);
		fill(membrane_potential.begin(), membrane_potential.end(), vars.memp_initval);
		fill(recovery_variable.begin(), recovery_variable.end(), vars.recv_initval);
		now_step = 0;
		total_step = (int)(vars.Tn/vars.dt);
		mt19937 rg(vars.rand_seed);
		random_generator = rg;
		continuation = -1;
		if (vars.expoTimeSeri) {
			ofstream clf_memp(vars.outfile_memp, ios::trunc | ios::binary);
			clf_memp.close();
		}
	}
	else if (mode == 1)		// continue mode
	{
		continuation = import_prev_vars(membrane_potential, recovery_variable, synaptic_current, vars);
		if (continuation == -1) {
			return throw_error("coding_error", "unexpected value for 'continuation'");
		}
		now_step = (int)(vars.Tn/vars.dt);
		mt19937 rg(vars.rand_seed);
		random_generator = rg;
		if (import_prev_spkt(spike_timestamps, vars.outfile_spkt) == EXIT_FAILURE) { return EXIT_FAILURE; }
		if (import_prev_spks(spike_timesteps, vars.outfile_spks) == EXIT_FAILURE) { return EXIT_FAILURE; }
		outlog << "> previous data are successfully imported\n\n"
			   << "> previous simulation duration T: " << vars.Tn << " ms\n"
			   << "> extend the duration to (in ms): ";
		double user_input;
		cin >> user_input;
		if (user_input <= vars.Tn) { return throw_error("invalid_input", to_string(user_input)); }
		outlog << '\n';
		vars.Tn = user_input;
		total_step = (int)(vars.Tn/vars.dt);
		if (import_adjm(synaptic_weights, vars) == EXIT_FAILURE) { return EXIT_FAILURE; }
		// Discard the first (T/dt) random numbers, so that the generation of random number begins properly
		int waste_step = 0;
		while (waste_step < now_step) {
			waste_step++;
			for (int node = 0; node < vars.mat_size; node++) {
				noise = norm_dist(random_generator);
			}
		}
	}

	neuron_type = vector<int>(vars.mat_size);
	if (classify_neuron_class(neuron_type, synaptic_weights) == EXIT_FAILURE) { return EXIT_FAILURE; }

	if (vars.suppr_lv != 0)
	{
		if (vars.suppr_type == -1) {
			if (vars.suppr_nodes.empty()) {
				suppress_inhibition(synaptic_weights, vars.suppr_lv);
			} else {
				suppress_inhibition_of_selected(synaptic_weights, inh_links, vars);
			}
		} else if (vars.suppr_type == 1) {
			if (vars.suppr_nodes.empty()) {
				suppress_excitation(synaptic_weights, vars.suppr_lv);
			} else {
				suppress_excitation_of_selected(synaptic_weights, exc_links, vars);
			}
		} else {
			return throw_error("coding_error", "unexpected value for 'vars.suppressed_link_type'");
		}
	}
	if (mode == 0) {
		if (vars.trunc_t_inh == -1) {					// use manual truncation time
			estimate_trunc_t(synaptic_weights, vars);	// else, estimate the spiking truncation time
		}
	}

	create_quick_link_ref(synaptic_weights, inh_links, exc_links);
    trunc_step_inh = (int)(vars.trunc_t_inh/vars.dt);
    trunc_step_exc = (int)(vars.trunc_t_exc/vars.dt);
	setup_hash_table_exp(HASHTABLE_EXP_INH, HASHTABLE_EXP_EXC, trunc_step_inh, trunc_step_exc);

	ofstream ofs_memp;
	if (vars.expoTimeSeri)
	{
		ofs_memp.open(vars.outfile_memp, ios::app | ios::binary);
		if (!ofs_memp.is_open()) { return throw_error("file_access", vars.outfile_memp); }
	}

	outlog << "[Initialization] completed in " << timer.stopwatch_lap()/1000 << " s\n\n";
	display_settings(vars);
	outlog << "[Computation] starts" << endl;

	TIMESERI_RESERVE_SIZE = ((size_t)(vars.TIMESERIES_BUFF / vars.mat_size) + 1) * vars.mat_size;
	expo_memp.reserve(TIMESERI_RESERVE_SIZE);
	if (vars.expoTimeSeri && (continuation == -1))
	{
		for (int i = 0; i < vars.mat_size; i++) {
			expo_memp.push_back(membrane_potential[i]);
		}
	}


	/* Main calculation loop */
	while (now_step < total_step)
	{
		now_step++;

		for (int node = 0; node < vars.mat_size; node++)
		{
			noise = vars.sigma * norm_dist(random_generator);

			memp_temp = membrane_potential[node];	// so that the other variables take
													// the membrane potential of the previous step
			membrane_potential[node] += (
				(0.04 * membrane_potential[node] * membrane_potential[node]) + (5 * membrane_potential[node])
				+ 140 - recovery_variable[node] + synaptic_current[node]
			) * vars.dt + noise * sqrt(vars.dt);

			if (neuron_type[node] == -1) {
				recovery_variable[node] += vars.a_inh * (
					vars.b_inh * memp_temp - recovery_variable[node]
				) * vars.dt;
			} else {
				recovery_variable[node] += vars.a_exc * (
					vars.b_exc * memp_temp - recovery_variable[node]
				) * vars.dt;
			}

			if (membrane_potential[node] >= 30) {
				if (neuron_type[node] == -1) {
					membrane_potential[node] = vars.c_inh;
					recovery_variable[node] += vars.d_inh;
				} else {
					membrane_potential[node] = vars.c_exc;
					recovery_variable[node] += vars.d_exc;
				}
				spike_timesteps[node].push_back(now_step);
				spike_timestamps[node].push_back((double)now_step*vars.dt);
			}

			conductance_inh = 0;
			for (auto &in_inh : inh_links[node]) {
				spike_sum = 0;
				for (int spk = spike_timesteps[in_inh].size()-1; spk >= 0; spk--) {
					diff_step = now_step - spike_timesteps[in_inh][spk];
					if (diff_step < trunc_step_inh) {
						spike_sum += HASHTABLE_EXP_INH[diff_step];
					} else { break; }
				}
				conductance_inh -= (synaptic_weights[node][in_inh] * spike_sum);
			}

			conductance_exc = 0;
			for (auto &in_exc : exc_links[node]) {
				spike_sum = 0;
				for (int spk = spike_timesteps[in_exc].size()-1; spk >= 0; spk--) {
					diff_step = now_step - spike_timesteps[in_exc][spk];
					if (diff_step < trunc_step_exc) {
						spike_sum += HASHTABLE_EXP_EXC[diff_step];
					} else { break; }
				}
				conductance_exc += (synaptic_weights[node][in_exc] * spike_sum);
			}

			synaptic_current[node] = vars.beta * (
				conductance_exc * (vars.thres_v_exc - memp_temp)
				- conductance_inh * (memp_temp - vars.thres_v_inh)
			);
		}

		if (vars.expoTimeSeri)    // Export time series data for all nodes (for >TIMESERIES_BUFF)
		{
			for (auto &v : membrane_potential) { expo_memp.push_back(v); }
			// Flush to output file and clear buffer
			if (expo_memp.size() >= vars.TIMESERIES_BUFF) {
				ofs_memp.write(reinterpret_cast<char*>(&expo_memp[0]), expo_memp.size()*sizeof(float));
				expo_memp.clear();
			}
		}
	}

	outlog << "[Computation] completed in " << timer.stopwatch_lap()/1000 << " s\n\n";
	outlog << "[Exportation] starts\n";

	export_info(continuation, timer.stopwatch_elapsed_t()/1000, vars);
	export_cont(continuation, membrane_potential, recovery_variable, synaptic_current, vars);
	export_spkt(spike_timestamps, vars.outfile_spkt);
	export_spks(spike_timesteps, vars.outfile_spks);

	/* Export time series data for all nodes (for <TIMESERIES_BUFF) */
	if (vars.expoTimeSeri) {
		if (vars.mat_size*vars.Tn/vars.dt < vars.TIMESERIES_BUFF) {
			if(export_time_series_bin(expo_memp, vars.outfile_memp, mode) == EXIT_FAILURE) {
				return EXIT_FAILURE;
			}
		} else {
			ofs_memp.write(reinterpret_cast<char*>(&expo_memp[0]), expo_memp.size()*sizeof(float));
		}
	}
	if (ofs_memp.is_open()) { ofs_memp.close(); }

	outlog << "[Exportation] completed in " << timer.stopwatch_lap()/1000 << " s\n\n";
	outlog << "COMPLETED :)" << endl;

	return EXIT_SUCCESS;
}

const int parser(int argc, char **argv, const char *optstr, Variables &vars)
{
	int c;
	while ((c = getopt(argc, (char**)argv, optstr)) != EOF)
	{
		switch (c)
		{
		case 'h':
			cout << "\nUsage:\n"
				 << "> start_simulation [-h] [-w W] [-i I] [-o O]\n\n";
			cout << "Brief:\n"
				 << "> Simulate the dynamics of a network of spiking neurons.\n\n";
			cout << "Input/output:\n"
				 << "> in:  [1] variable file          | vars.txt\n"
				 << ">      [2] synaptic weight file   | DIVxx_gji\n"
				 << "> out: [1] simulation information | info.txt\n"
				 << ">      [2] spiking time-stamps    | spkt.txt\n"
				 << ">      [3] potential time series  | memp.dat\n"
				 << ">      [4] continuation file      | cont.dat\n\n";
			cout << "Optional arguments:\n"
				 << "  -w <str>\tpath to synaptic weight file, default: using [vars]\n"
				 << "  -i <str>\tpath to input directory, default: current\n"
				 << "  -o <str>\tpath to output directory, default: current\n"
				//  << "  -l <str>\tlog file name (not for user)\n"
				 << "  -h\t\tshow this help message and exit\n\n";
			return EXIT_FAILURE;
			break;
		case 'w':
			vars.inadjm_path = optarg;
			break;
		case 'i':
			vars.infile_path = optarg;
			break;
		case 'o':
			vars.outfile_path = optarg;
			break;
		case 'l':
			vars.logfile = optarg;
			break;
		case '?':
			// cout << "\nInvalidOption: " << (char)optopt << ", use [-h] flag to seek help.\n\n";
			cout << "\nUsage:\n"
				 << "  start_simulation [-h] [-w W] [-i I] [-o O]\n\n";
			cout << "Brief:\n"
				 << "  simulate the dynamics of a network of spiking neurons.\n\n";
			cout << "Input/output:\n"
				 << "  in:  [1] variable file          | vars.txt\n"
				 << "       [2] synaptic weight file   | DIVxx_gji\n"
				 << "  out: [1] simulation information | info.txt\n"
				 << "       [2] spiking time-stamps    | spkt.txt\n"
				 << "       [3] potential time series  | memp.dat\n"
				 << "       [4] continuation file      | cont.dat\n\n";
			cout << "Optional arguments:\n"
				 << "  -w <str>\tpath to synaptic weight file, default: using [vars]\n"
				 << "  -i <str>\tpath to input directory, default: current\n"
				 << "  -o <str>\tpath to output directory, default: current\n"
				//  << "  -l <str>\tlog file name (not for user)\n"
				 << "  -h\t\tshow this help message and exit\n\n";
			return EXIT_FAILURE;
			break;
		default:
			cout << "\nUnknownError: no handler for option " << (char)c << " or " << c << "\n\n";
			return EXIT_FAILURE;
			break;
		}
	}
	return EXIT_SUCCESS;
}

path set_logfile_name(Variables &vars, string time_fmt2)
{
	string log_fname("log_"+time_fmt2+".txt");
	path logfile_path;
	if (vars.logfile.string() == "") {
		if (vars.outfile_path.string() == "") {
			fs::create_directory("log");
		} else {
			fs::create_directory(vars.outfile_path);
			fs::create_directory(vars.outfile_path / "log");
			cout << "ok";
		}
		logfile_path = vars.outfile_path / "log" / log_fname;
	} else {
		logfile_path = vars.logfile;
	}
	return logfile_path;
}

void set_io_path(Variables &vars)
{
	vars.infile_vars  = vars.infile_path / vars.infile_vars;
	vars.outfile_info = vars.outfile_path / vars.outfile_info;
	vars.outfile_spkt = vars.outfile_path / vars.outfile_spkt;
	vars.outfile_spks = vars.outfile_path / vars.outfile_spks;
	vars.outfile_memp = vars.outfile_path / vars.outfile_memp;
	vars.outfile_curr = vars.outfile_path / vars.outfile_curr;
	vars.outfile_recv = vars.outfile_path / vars.outfile_recv;
	vars.outfile_cont = vars.outfile_path / vars.outfile_cont;
}

bool check_file_existance(Variables &vars)
{
	ifstream ifs;
	ifs.open(vars.outfile_info);
	bool infoExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_spkt);
	bool spktExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_spks);
	bool spksExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_curr);
	bool currExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_memp);
	bool mempExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_recv);
	bool recvExist = ifs.good();
	ifs.close();
	ifs.open(vars.outfile_cont);
	bool contExist = ifs.good();
	ifs.close();

	return (
		infoExist || spktExist || spksExist || currExist || 
		mempExist || recvExist || contExist
	);
}

void display_settings(Variables &vars)
{
	outlog << "- Network: " << vars.infile_adjm << '\n'
		   << "- T:  " << vars.Tn << " ms\n"
		   << "- dt: " << vars.dt << " ms\n"
		   << "- strength of noise (sigma): " << vars.sigma << '\n'
		   << "- seed to random noise gen:  " << vars.rand_seed << '\n';
	if (vars.suppr_lv != 0) {
		outlog << "- suppression level:    " << vars.suppr_lv << '\n'
			   << "- suppressed link type: ";
		if (vars.suppr_type == -1) { outlog << "inh\n"; }
		else { outlog << "exc\n"; }
	}
	outlog << "- export time series:   ";
	if (vars.expoTimeSeri == true) { outlog << "yes\n\n"; }
	else { outlog << "no\n\n"; }
}

string remove_whitespace(string &str)
{
	str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
	return str;
}

const int import_vars(Variables &vars)
{
	vector<string> variables;
	string line, val;
	ifstream ifs(vars.infile_vars);
	if (ifs.is_open()) {
		while (getline(ifs, line, '\n')) {
			if (line.find('=') != string::npos) {
				stringstream ss(line);
				getline(ss, val, '=');
				getline(ss, val, '=');
				variables.push_back(remove_whitespace(val));
			}
		}
		ifs.close();
	} else { return throw_error("vars_missing", vars.infile_vars); }

	vars.infile_adjm = variables[0];
	if (variables[1] == "full") { vars.mat_format = "full"; }
	else if (variables[1] == "nonzero") { vars.mat_format = "nonzero"; }
	else {
		string err_msg[3] = {variables[1], vars.mat_format, "matrix format (full/nonzero)"};
		return throw_error("invalid_input_file", err_msg);
	}
	vars.mat_size = stoi(variables[2]);
	if (variables[3] == "tab") { vars.delim = '\t'; }
	else if (variables[3] == "space") { vars.delim = ' '; }
	else { vars.delim = variables[3][0]; }
	vars.dt = stod(variables[4]);
	vars.Tn = stod(variables[5]);
	vars.sigma = stod(variables[6]);
	vars.beta = stod(variables[7]);
	vars.rand_seed = stof(variables[8]);
	vars.trunc_t_inh = (variables[9] == "auto") ? -1 : stod(variables[9]);
	vars.trunc_t_exc = (variables[9] == "auto") ? -1 : stod(variables[9]);
	vars.suppr_lv = stod(variables[10]);
	if (variables[11] == "inh") { vars.suppr_type = -1; }
	else if (variables[11] == "exc") { vars.suppr_type = 1; }
	else {
		string err_msg[3] = {variables[11], vars.infile_vars.string(), "type of links (inh/exc)"};
		return throw_error("invalid_input_file", err_msg);
	}
	string node, suppr_nodes = variables[12];
	stringstream ss(suppr_nodes);
	vars.suppr_nodes.clear();
	while (getline(ss, node, ',')) { vars.suppr_nodes.push_back(stoi(remove_whitespace(node))-1); }
	vars.a_inh = stod(variables[13]);
	vars.b_inh = stod(variables[14]);
	vars.c_inh = stod(variables[15]);
	vars.d_inh = stod(variables[16]);
	vars.a_exc = stod(variables[17]);
	vars.b_exc = stod(variables[18]);
	vars.c_exc = stod(variables[19]);
	vars.d_exc = stod(variables[20]);
	vars.thres_v_inh = stod(variables[21]);
	vars.thres_v_exc = stod(variables[22]);
	vars.tau_inh = stod(variables[23]);
	vars.tau_exc = stod(variables[24]);
	vars.memp_initval = stod(variables[25]);
	vars.recv_initval = stod(variables[26]);
	vars.expoTimeSeri = (variables[27] == "true") ? true : false;
	return EXIT_SUCCESS;
}

const int import_adjm(vector<vector<double>> &synaptic_weights, Variables &vars)
{
	vector<vector<double>> adjm_temp;
	vector<double> row_buf;
	string line, elem;
	path adjm_path = fs::current_path() / vars.inadjm_path / vars.infile_adjm;
	ifstream ifs(adjm_path);
	if (ifs.is_open()) {
		if (vars.mat_format == "nonzero") {
			synaptic_weights = vector<vector<double>>(vars.mat_size, vector<double>(vars.mat_size, 0));
			while(getline(ifs, line, '\n')) {
				stringstream ss(line);
				while(getline(ss, elem, vars.delim)) {
					if (elem != "") { row_buf.push_back(stof(remove_whitespace(elem))); }
				}
				adjm_temp.push_back(row_buf);
				row_buf.clear();
			}
			for (size_t i = 0; i < adjm_temp.size(); i++) {
				synaptic_weights[(int)adjm_temp[i][1]-1][(int)adjm_temp[i][0]-1] = adjm_temp[i][2];
			}
		} else if (vars.mat_format == "full") {
			while(getline(ifs, line, '\n')) {
				stringstream ss(line);
				while(getline(ss, elem, vars.delim)) { row_buf.push_back(stof(elem)); }
				adjm_temp.push_back(row_buf);
				row_buf.clear();
			}
			synaptic_weights = adjm_temp;
			if (synaptic_weights.size() < 2) { return throw_error("matrix_size"); }
		} else {
			return throw_error("coding_error", "unexpected value for 'vars.mat_format'");
		}
		ifs.close();
	} else { return throw_error("file_access", adjm_path); }
	return EXIT_SUCCESS;
}

const int estimate_trunc_t(vector<vector<double>> &synaptic_weights, Variables &vars)
{
	double w_inh = 0, w_inh_max = 0, w_exc = 0, w_exc_max = 0;
	for (size_t i = 0; i < synaptic_weights.size(); i++) {
		for (size_t j = 0; j < synaptic_weights[i].size(); j++) {
			if (synaptic_weights[i][j] < 0) { w_inh += synaptic_weights[i][j]; }
			else if (synaptic_weights[i][j] > 0) { w_exc += synaptic_weights[i][j]; }
		}
		if (abs(w_inh) > w_inh_max) { w_inh_max = abs(w_inh); }
		if (abs(w_exc) > w_exc_max) { w_exc_max = abs(w_exc); }
	}
	vars.trunc_t_inh = vars.tau_inh * log(vars.prec * vars.beta * w_inh_max);
	vars.trunc_t_exc = vars.tau_exc * log(vars.prec * vars.beta * w_exc_max);
	if (w_inh_max == 0 || vars.beta == 0) { vars.trunc_t_inh = 0; }
	if (w_exc_max == 0 || vars.beta == 0) { vars.trunc_t_exc = 0; }
	return EXIT_SUCCESS;
}

int import_prev_vars(vector<double> &memp, vector<double> &recv, vector<double> &curr, Variables &vars)
{
	int continuation = -1;
	vector<string> prev_vars;
	string line, val;
	stringstream ss;
	ifstream ifs(vars.outfile_cont);
	if (ifs.is_open()) {
		getline(ifs, line, '\n');
		ss.str(line);
		getline(ss, val, '|');
		continuation = stoi(val); // import the number of times of continuation (continuation)
		getline(ss, val, '|');
		vars.expoTimeSeri = stoi(val); // import (bool) of whether time series will be exported

		getline(ifs, line, '\n');
		ss.clear();
		ss.str(line);
		while (getline(ss, val, '|')) {
			prev_vars.push_back(val); // import all variables and parameters from previous calculation, e.g., dt, sigma, etc.
		}
		getline(ifs, line, '\n');
		ss.clear();
		ss.str(line);
		while (getline(ss, val, '\t')) {
			memp.push_back(stod(val)); // import membrane potentials of all neurons at the last time step from previous calculation
		}
		getline(ifs, line, '\n');
		ss.clear();
		ss.str(line);
		while (getline(ss, val, '\t')) {
			recv.push_back(stod(val)); // import recovery variables of all neurons at the last time step from previous calculation
		}
		getline(ifs, line, '\n');
		ss.clear();
		ss.str(line);
		while (getline(ss, val, '\t')) {
			curr.push_back(stod(val)); // import synaptic current of all neurons at the last time step from previous calculation
		}
		ifs.close();
	} else {
		throw_error("file_access", vars.outfile_cont);
		return -1;
	}
	vars.infile_adjm    = prev_vars[25];
	vars.mat_format     = prev_vars[26];
	vars.delim          = prev_vars[27][0];
	vars.mat_size       = stoi(prev_vars[0]);
	vars.dt             = stod(prev_vars[1]);
	vars.Tn             = stod(prev_vars[2]);
	vars.rand_seed      = stof(prev_vars[3]);
	vars.a_inh          = stod(prev_vars[4]);
	vars.b_inh          = stod(prev_vars[5]);
	vars.c_inh          = stod(prev_vars[6]);
	vars.d_inh          = stod(prev_vars[7]);
	vars.a_exc          = stod(prev_vars[8]);
	vars.b_exc          = stod(prev_vars[9]);
	vars.c_exc          = stod(prev_vars[10]);
	vars.d_exc          = stod(prev_vars[11]);
	vars.sigma          = stod(prev_vars[12]);
	vars.thres_v_inh    = stod(prev_vars[13]);
	vars.thres_v_exc    = stod(prev_vars[14]);
	vars.tau_inh        = stod(prev_vars[15]);
	vars.tau_exc        = stod(prev_vars[16]);
	vars.beta           = stod(prev_vars[17]);
	vars.memp_initval   = stod(prev_vars[18]);
	vars.recv_initval   = stod(prev_vars[19]);
	vars.trunc_t_inh    = stod(prev_vars[20]);
	vars.trunc_t_exc    = stod(prev_vars[21]);
	vars.suppr_lv       = stod(prev_vars[22]);
	vars.suppr_type     = stoi(prev_vars[23]);
	string node, suppr_nodes = prev_vars[24];
	ss.clear();
	ss.str(suppr_nodes);
	vars.suppr_nodes.clear();
	while (getline(ss, node, ',')) { vars.suppr_nodes.push_back(stoi(node)); }
	// vars.outfile_spkt   = prev_vars[28];
	// vars.outfile_curr   = prev_vars[29];
	// vars.outfile_memp   = prev_vars[30];
	// vars.outfile_recv   = prev_vars[31];
	// vars.outfile_info   = prev_vars[32];
	return continuation;
}

const int import_prev_spkt(vector<vector<double>> &spkt, path &outfile_spkt)
{
	vector<double> row_buf;
	string line, elem;
	ifstream ifs(outfile_spkt);
	if (ifs.is_open()) {
		while(getline(ifs, line, '\n')) {
			stringstream ss(line);
			while(getline(ss, elem, '\t')) { row_buf.push_back(stof(elem)); }
			row_buf.erase(row_buf.begin()); // delete the number of spikes
			spkt.push_back(row_buf);
			row_buf.clear();
		}
		ifs.close();
	} else { return throw_error("file_access", outfile_spkt); }
	return EXIT_SUCCESS;
}

const int import_prev_spks(vector<vector<int>> &spks, path &outfile_spks)
{
	vector<int> row_buf;
	string line, elem;
	ifstream ifs(outfile_spks);
	if (ifs.is_open()) {
		while(getline(ifs, line, '\n')) {
			stringstream ss(line);
			while(getline(ss, elem, '\t')) { row_buf.push_back(stoi(elem)); }
			row_buf.erase(row_buf.begin()); // delete the number of spikes
			spks.push_back(row_buf);
			row_buf.clear();
		}
		ifs.close();
	} else { return throw_error("file_access", outfile_spks); }
	return EXIT_SUCCESS;
}

const int classify_neuron_class(vector<int> &neuron_type, vector<vector<double>> &synaptic_weights)
{
	for (size_t i = 0; i < synaptic_weights.size(); i++) {
		for (size_t j = 0; j < synaptic_weights.size(); j++) {
			if (synaptic_weights[i][j] > 0) {
				if (neuron_type[j] == -1) {
					int current_loc[2] = {(int)(i+1), (int)(j+1)};
					return throw_error("neuron_type", current_loc);
				}
				neuron_type[j] = 1;
			}
			else if (synaptic_weights[i][j] < 0) {
				if (neuron_type[j] == 1) {
					int current_loc[2] = {(int)(i+1), (int)(j+1)};
					return throw_error("neuron_type", current_loc);
				}
				neuron_type[j] = -1;
			}
		}
	}
	return EXIT_SUCCESS;
}

const int create_quick_link_ref(vector<vector<double>> &synaptic_weights,
								vector<vector<int>> &inh_links, vector<vector<int>> &exc_links)
{
	vector<int> inh_temp, exc_temp;
	for (size_t i = 0; i < synaptic_weights.size(); i++) {
		for (size_t j = 0; j < synaptic_weights.size(); j++) {
			if (synaptic_weights[i][j] < 0) { inh_temp.push_back(j); }
			else if (synaptic_weights[i][j] > 0) { exc_temp.push_back(j); }
		}
		inh_links.push_back(inh_temp);
		exc_links.push_back(exc_temp);
		inh_temp.clear();
		exc_temp.clear();
	}
	return EXIT_SUCCESS;
}

void setup_hash_table_exp(vector<double> &HASHTABLE_EXP_INH, vector<double> &HASHTABLE_EXP_EXC,
						  int trunc_step_inh, int trunc_step_exc)
{
	HASHTABLE_EXP_INH = vector<double>(trunc_step_inh+1);
    HASHTABLE_EXP_EXC = vector<double>(trunc_step_exc+1);
    for (int i = 0; i < trunc_step_inh+1; i++) {
        HASHTABLE_EXP_INH[i] = exp(-i * vars.dt / vars.tau_inh);
    }
    for (int i = 0; i < trunc_step_exc+1; i++) {
        HASHTABLE_EXP_EXC[i] = exp(-i * vars.dt / vars.tau_exc);
    }
}

double calculate_standard_deviation_neg(vector<vector<double>> &synaptic_weights)
{
	int count = 0;
	double sum = 0.0, mean, sd = 0.0;
	vector<double> neg_weights;
	for (auto &row : synaptic_weights) {
		for (auto const &elem : row) { if (elem < 0) { sum += elem; ++count; } }
	}
	mean = sum / count;
	for (auto &row : synaptic_weights) {
		for (auto elem : row) { if (elem < 0) { sd += pow(elem - mean, 2); } }
	}
	return sqrt(sd / count);
}

double calculate_standard_deviation_pos(vector<vector<double>> &synaptic_weights)
{
	int count = 0;
	double sum = 0.0, mean, sd = 0.0;
	vector<double> neg_weights;
	for (auto &row : synaptic_weights) {
		for (auto const &elem : row) { if (elem > 0) { sum += elem; ++count; } }
	}
	mean = sum / count;
	for (auto &row : synaptic_weights) {
		for (auto elem : row) { if (elem > 0) { sd += pow(elem - mean, 2); } }
	}
	return sqrt(sd / count);
}

void suppress_inhibition(vector<vector<double>>& synaptic_weights, double suppr_lv)
{
	double sd_inh = calculate_standard_deviation_neg(synaptic_weights);

	for (auto &row : synaptic_weights) {
		for (auto &elem : row) {
			if (elem < 0) {
				elem += sd_inh * suppr_lv;
				if (elem > 0) { elem = 0; }
			}
		}
	}
}

void suppress_inhibition_of_selected(vector<vector<double>> &synaptic_weights,
									 vector<vector<int>> &inh_links, Variables &vars)
// Suppress inhibition incoming links for each neuron in 'node_idx'
{
	double sd_inh = calculate_standard_deviation_neg(synaptic_weights);

	for (auto &selected : vars.suppr_nodes) {
		for (auto &link : inh_links[selected]) {
			synaptic_weights[selected][link] += sd_inh * vars.suppr_lv;
			if (synaptic_weights[selected][link] > 0) { synaptic_weights[selected][link] = 0; }
		}
	}
}

void suppress_excitation(vector<vector<double>> &synaptic_weights, double suppr_lv)
{
	double sd_exc = calculate_standard_deviation_pos(synaptic_weights);

	for (auto &row : synaptic_weights) {
		for (auto &elem : row) {
			if (elem > 0) {
				elem -= sd_exc * suppr_lv;
				if (elem < 0) { elem = 0; }
			}
		}
	}
}

void suppress_excitation_of_selected(vector<vector<double>> &synaptic_weights,
									 vector<vector<int>> &exc_links, Variables &vars)
// Suppress excitatory incoming links for each neuron in 'node_idx'
{
	double sd_exc = calculate_standard_deviation_pos(synaptic_weights);

	for (auto &selected : vars.suppr_nodes) {
		for (auto &link : exc_links[selected]) {
			synaptic_weights[selected][link] -= sd_exc * vars.suppr_lv;
			if (synaptic_weights[selected][link] < 0) { synaptic_weights[selected][link] = 0; }
		}
	}
}

const int export_info(int continuation, float time_elapsed, Variables &vars)
{
	auto [time_fmt1, _] = timer.report_time();
	if (continuation == -1) {
		ofstream ofs(vars.outfile_info, ios::trunc);
		if (ofs.is_open()) {
			ofs << code_ver << '\n'
				<< "--------------------------------------------------\n"
				<< "computation finished at: " << time_fmt1
				<< "time elapsed: " << time_elapsed << " s\n\n"
				<< "[Numerical Settings]" << '\n'
				<< "network file name:\t\t\t" << vars.infile_adjm << '\n'
				<< "number of neurons (N):\t\t" << vars.mat_size << '\n'
				<< "time step size (dt):\t\t" << vars.dt << '\n'
				<< "simulation duration (T):\t" << vars.Tn << '\n'
				<< "random seed number:\t\t\t" << vars.rand_seed << '\n'
				<< "white noise S.D. (sigma):\t" << vars.sigma << '\n'
				<< "matrix parameter (beta):\t" << vars.beta << '\n'
				<< "spike truncation time:\t\t" << vars.trunc_t_inh << " (inh)\n"
				<< "spike truncation time:\t\t" << vars.trunc_t_exc << " (exc)\n\n"
				<< "[Suppression of Synaptic Weights]\n"
				<< "suppression level:\t\t\t" << vars.suppr_lv << '\n';
			string type_of_links = vars.suppr_type == 1 ? "exc" : "inh";
			ofs << "type of links:\t\t\t\t" << type_of_links << '\n';
			if (vars.suppr_nodes.empty() == false) {
				ofs << "suppressed nodes:\t\t\t" << vars.suppr_nodes[0];
				for (size_t i = 1; i < vars.suppr_nodes.size(); i++) { ofs << ", " << vars.suppr_nodes[i]; }
				ofs << '\n';
			}
			ofs << "\n[Initial Values]\n"
				<< "membrane potential:\t\t\t" << vars.memp_initval << '\n'
				<< "recovery variable:\t\t\t" << vars.recv_initval << '\n' << endl;
			ofs.close();
		} else { return EXIT_FAILURE; }
	} else {
		ofstream ofs(vars.outfile_info, ios::app);
		if (ofs.is_open()) {
			ofs << "--------------------------------------------------\n"
				<< "computation finished at: " << time_fmt1
				<< "time elapsed: " << time_elapsed << "\n\n"
				<< "extend duration (T) to:\t\t" << vars.Tn << "\n\n";
			ofs.close();
		} else { return EXIT_FAILURE; }
	}
	return EXIT_SUCCESS;
}

const int export_cont(int continuation, vector<double> &memp, vector<double> &recv,
					  vector<double> &curr, Variables &vars)
{
	ofstream ofs(vars.outfile_cont, ios::trunc);
	ofs.precision(numeric_limits<double>::max_digits10);
	if (ofs.is_open()) {
		ofs << ++continuation << '|' << vars.expoTimeSeri << '\n';

		ofs << vars.mat_size << '|' << vars.dt << '|' << vars.Tn << '|' << vars.rand_seed << '|'
			<< vars.a_inh << '|' << vars.b_inh << '|' << vars.c_inh << '|' << vars.d_inh << '|'
			<< vars.a_exc << '|' << vars.b_exc << '|' << vars.c_exc << '|' << vars.d_exc << '|'
			<< vars.sigma << '|'
			<< vars.thres_v_inh << '|' << vars.thres_v_exc << '|'
			<< vars.tau_inh << '|' << vars.tau_exc << '|'
			<< vars.beta << '|'
			<< vars.memp_initval << '|' << vars.recv_initval << '|'
			<< vars.trunc_t_inh << '|' << vars.trunc_t_exc << '|'
			<< vars.suppr_lv << '|' << vars.suppr_type << '|';
		if (vars.suppr_nodes.empty() == false) {
			ofs << vars.suppr_nodes[0];
			for (size_t i = 1; i < vars.suppr_nodes.size(); i++) { ofs << ',' << vars.suppr_nodes[i]; }
		}
		ofs << '|' << vars.infile_adjm.string() << '|' << vars.mat_format << '|' << vars.delim << '|'
			<< vars.outfile_info.string() << '|' << vars.outfile_spks.string() << '|'
			<< vars.outfile_spkt.string() << '|' << vars.outfile_memp.string() << '|'
			<< vars.outfile_curr.string() << '|' << vars.outfile_recv.string() << endl;

		ofs << memp[0];
		for (int i = 1; i < vars.mat_size; i++) { ofs << '\t' << memp[i]; }
		ofs << '\n' << recv[0];
		for (int i = 1; i < vars.mat_size; i++) { ofs << '\t' << recv[i]; }
		ofs << '\n' << curr[0];
		for (int i = 1; i < vars.mat_size; i++) { ofs << '\t' << curr[i]; }
		ofs.close();
	} else { return EXIT_FAILURE; }
	return EXIT_SUCCESS;
}

const int export_spkt(vector<vector<double>> &spkt, path &outfile)
{
	char delimiter = '\t';
	ofstream ofs(outfile, ios::trunc);
	if (ofs.is_open()) {
		ofs << spkt[0].size();
		for (auto &elem : spkt[0]) { ofs << delimiter << elem; }
		for (size_t row = 1; row < spkt.size(); row++) {
			ofs << '\n' << spkt[row].size();
			for (auto &elem : spkt[row]) { ofs << delimiter << elem; }
		}
		ofs.close();
	} else { return EXIT_FAILURE; }
	return EXIT_SUCCESS;
}

const int export_spks(vector<vector<int>> &spks, path &outfile)
{
	char delimiter = '\t';
	ofstream ofs(outfile, ios::trunc);
	if (ofs.is_open()) {
		ofs << spks[0].size();
		for (auto &elem : spks[0]) { ofs << delimiter << elem; }
		for (size_t row = 1; row < spks.size(); row++) {
			ofs << '\n' << spks[row].size();
			for (auto &elem : spks[row]) { ofs << delimiter << elem; }
		}
		ofs.close();
	} else { return EXIT_FAILURE; }
	return EXIT_SUCCESS;
}

const int export_time_series_bin(vector<float> &data_array, path &outfile, int output_mode)
{
	if (output_mode == 0) {
		ofstream ofs(outfile, ios::trunc | ios::binary);
		ofs.write(reinterpret_cast<char*>(&data_array[0]), data_array.size()*sizeof(float));
		ofs.close();
	} else if (output_mode == 1) {
		ofstream ofs(outfile, ios::app | ios::binary);
		ofs.write(reinterpret_cast<char*>(&data_array[0]), data_array.size()*sizeof(float));
		ofs.close();
	} else {
		string err_msg = "invalid 'output_mode'";
		return throw_error("coding_error", &err_msg);
	}
	return EXIT_SUCCESS;
}

const int throw_error(string err_type)
{
	if (err_type == "user_terminate") {
		outlog << "\n<Program terminated by user>" << endl;
	} else if (err_type == "matrix_size") {
		outlog << "\n<InvalidMatrixSize> the input synaptic weights matrix is smaller than 2x2" << endl;
	} else {
		outlog << "\nunknown failure [1]" << endl;
	}
	return EXIT_FAILURE;
}

const int throw_error(string err_type, path &err_path)
{
	string msg;
	if (err_type == "file_access") {
		msg = "<FileAccessError> \""+err_path.string()+"\" cannot be accessed or it does not exist";
	} else if (err_type == "vars_missing") {
		msg = "<InputFileNotFound> variable file \""+err_path.string()+"\" cannot be found";
	} else {
		msg = "unknown failure [2]";
	}
	outlog << "\n" << msg << endl;
	return EXIT_FAILURE;
}

const int throw_error(string err_type, string err_msg)
{
	string msg;
	if (err_type == "invalid_input") {
		msg = "<InvalidInput> ["+err_msg+"] is invalid";
	} else if (err_type == "coding_error") {
		msg = "<CodingError> " + err_msg;
	} else {
		msg = "unknown failure [3]";
	}
	outlog << "\n" << msg << endl;
	return EXIT_FAILURE;
}

const int throw_error(string err_type, string *err_msg)
{
	string msg;
	if (err_type == "invalid_input_file") {
		msg = "<InvalidInput> ["+*err_msg+"] is invalid in file \""+err_msg[1]+" > "
			  +err_msg[2]+"\"";
	} else if (err_type == "invalid_input") {
		msg = "<InvalidInput> ["+*err_msg+"] is invalid";
	} else if (err_type == "coding_error") {
		msg = "<CodingError> " + *err_msg;
	} else {
		msg = "unknown failure [4]";
	}
	outlog << "\n" << msg << endl;
	return EXIT_FAILURE;
}

const int throw_error(string err_type, int *err_num)
{
	string msg;
	if (err_type == "invalid_main_arg") {
		msg = "<InvalidOption> unexpected arguments for start_simulation::main(): it takes 0 or 2 or 3 arguments, but "
			  +to_string(err_num[0]-1)+" was(were) given\n"
			  +"- 2 args: [matrix file directory]  [I/O directory]\n"
			  +"- 3 args: [matrix file directory]  [I/O directory]  [log file path]";
	} else if (err_type == "neuron_type") {
		msg = "<Error> neuron classification: inconsistent neuron class is detected at ("
			  +to_string(err_num[0])+", "+to_string(err_num[1])+")";
	} else {
		msg = "unknown failure [5]";
	}
	outlog << "\n" << msg << endl;
	return EXIT_FAILURE;
}

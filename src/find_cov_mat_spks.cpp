#include <filesystem>
// #include <algorithm>
// #include <unistd.h>
#include <iostream>
// #include <fstream>
// #include <sstream>
// #include <numeric>
// #include <string>
// #include <limits>
// #include <chrono>
// #include <random>
#include <vector>
// #include <tuple>
// #include <cmath>
#include "myinc.h"
#define _CRT_SECURE_NO_WARNINGS

using namespace std;
using namespace myinc;
namespace fs = std::filesystem;
using path = fs::path;
using fs::current_path;

struct Vars
{	/* Settings */
	double	coef	= 50000;

    size_t	N		= 4095;
    double	tau		= 0.05;
    double	Tn		= 10000;
    double	dt		= 0.05;
    path	infile	= "spkt.txt";
    path	outdir	= "";
	// string	logfile	= "";
	// bool	gotTau	= false;
	// bool	outK0	= false;
	// bool	outKT	= false;
} vars;

const int import_spkt(vector<vector<double>>&, path&);

int main() {

	timer.stopwatch_begin();

	double sum_s_ij;
	int total_step = vars.Tn / vars.dt;
	vector<vector<double>> covar_mat(vars.N, vector<double>(vars.N));
    vector<vector<double>> spkt, spkt_lag; // Record the time step of spikes
    vector<double> avg_s(vars.N);
    import_spkt(spkt, vars.infile);

	// Calculate equal-time covariance matrix K_ij(0)
	cout << "Finding equal-time covariance matrix K(0)\n";
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
			avg_s[i] = (double)spkt[i].size() / total_step;
	}
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
		for (size_t j = 0; j <= i; j++) {
			sum_s_ij = 0;
			if (spkt[j].size() > spkt[i].size()) {
				for (auto &spk : spkt[i]) {
					sum_s_ij += binary_search(spkt[j].begin(), spkt[j].end(), spk);
				}
			} else {
				for (auto &spk : spkt[j]) {
					sum_s_ij += binary_search(spkt[i].begin(), spkt[i].end(), spk);
				}
			}
			covar_mat[i][j] = vars.coef*( (double)sum_s_ij/total_step - avg_s[i]*avg_s[j] );
		}
	}
	// K(0) is symmetric
	#pragma omp parallel for
	for (size_t j = 0; j < vars.N; j++) {
		for (size_t i = 0; i < j; i++) {
			covar_mat[i][j] = covar_mat[j][i];
		}
	}
	path output_path = vars.outdir / "K_0";
	if (fwrite_dense_matrix(covar_mat, output_path) == EXIT_SUCCESS) {
		cout << "Equal-time covariance matrix K(0) written into file \"" << output_path.string() << "\"\n";
	}
	vector<vector<double>>(vars.N, vector<double>(vars.N, 0)).swap(covar_mat);
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";

	// Calculate time-lagged covariance matrix K_ij(tau)
	cout << "Finding equal-time covariance matrix K(tau)\n";
	spkt_lag = spkt;
	for (size_t i = 0; i < vars.N; i++) {
		for (auto &t : spkt_lag[i]) {
			t += vars.tau;
		}
	}
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
		for (size_t j = 0; j < vars.N; j++) {
			sum_s_ij = 0;
			if (spkt_lag[i].size() < spkt[j].size()) {
				for (auto &spk : spkt_lag[i]) {
					sum_s_ij += binary_search(spkt[j].begin(), spkt[j].end(), spk);
				}
			} else {
				for (auto &spk : spkt[j]) {
					sum_s_ij += binary_search(spkt_lag[i].begin(), spkt_lag[i].end(), spk);
				}
			}
			covar_mat[i][j] = vars.coef*( (double)sum_s_ij/total_step - avg_s[i]*avg_s[j] );
		}
	}
	output_path = vars.outdir / "K_tau";
	if (fwrite_dense_matrix(covar_mat, output_path) == EXIT_SUCCESS) {
		cout << "Equal-time covariance matrix K(tau) written into file \"" << output_path.string() << "\"\n";
	}
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";
}

const int import_spkt(vector<vector<double>> &spkt, path &outfile_spkt)
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
	} else { cout << "error in opening file."; }
	return EXIT_SUCCESS;
}

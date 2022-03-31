#include <filesystem>
#include <iostream>
#include <vector>
#include "myinc.h"
#define _CRT_SECURE_NO_WARNINGS

using namespace std;
using namespace myinc;
using path = std::filesystem::path;

struct Vars
{	/* Settings */
	double	coef	= 1;

    size_t	N		= 4095;
    double	tau		= 0.05;
    double	Tn		= 10000;
    double	dt		= 0.05;
    path	infile	= "spks.txt";
    path	outdir	= "matrices";
	// string	logfile	= "";
	// bool	gotTau	= false;
	// bool	outK0	= false;
	// bool	outKT	= false;
} vars;

const int import_spks(vector<vector<int>>&, path&);

int main() {

	timer.stopwatch_begin();

	double sum_s_ij;
	int total_step = vars.Tn / vars.dt, lag_s = vars.tau / vars.dt;
	vector<vector<double>> cov_mat(vars.N, vector<double>(vars.N));
    vector<vector<int>> spks, spks_lag; // Record the time step of spikes
    vector<double> avg_s(vars.N);
    import_spks(spks, vars.infile);

	// Calculate equal-time covariance matrix K_ij(0)
	cout << "Finding equal-time covariance matrix K(0)\n";
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
			avg_s[i] = (double)spks[i].size() / total_step;
	}
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
		for (size_t j = 0; j <= i; j++) {
			sum_s_ij = 0;
			if (spks[j].size() > spks[i].size()) {
				for (auto &spk : spks[i]) {
					sum_s_ij += binary_search(spks[j].begin(), spks[j].end(), spk);
				}
			} else {
				for (auto &spk : spks[j]) {
					sum_s_ij += binary_search(spks[i].begin(), spks[i].end(), spk);
				}
			}
			cov_mat[i][j] = vars.coef*( (double)sum_s_ij/total_step - avg_s[i]*avg_s[j] );
		}
	}
	// K(0) is symmetric
	#pragma omp parallel for
	for (size_t j = 0; j < vars.N; j++) {
		for (size_t i = 0; i < j; i++) {
			cov_mat[i][j] = cov_mat[j][i];
		}
	}
	path output_path = vars.outdir / "K_0";
	if (fwrite_dense_matrix(cov_mat, output_path) == EXIT_SUCCESS) {
		cout << "Equal-time covariance matrix K(0) written into file \"" << output_path.string() << "\"\n";
	}
	vector<vector<double>>(vars.N, vector<double>(vars.N, 0)).swap(cov_mat);
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";

	// Calculate time-lagged covariance matrix K_ij(tau)
	cout << "Finding equal-time covariance matrix K(tau)\n";
	spks_lag = spks;
	for (size_t i = 0; i < vars.N; i++) {
		for (auto &s : spks_lag[i]) {
			s += lag_s;
		}
	}
	#pragma omp parallel for
	for (size_t i = 0; i < vars.N; i++) {
		for (size_t j = 0; j < vars.N; j++) {
			sum_s_ij = 0;
			if (spks_lag[i].size() < spks[j].size()) {
				for (auto &spk : spks_lag[i]) {
					sum_s_ij += binary_search(spks[j].begin(), spks[j].end(), spk);
				}
			} else {
				for (auto &spk : spks[j]) {
					sum_s_ij += binary_search(spks_lag[i].begin(), spks_lag[i].end(), spk);
				}
			}
			cov_mat[i][j] = vars.coef*( (double)sum_s_ij/total_step - avg_s[i]*avg_s[j] );
		}
	}
	output_path = vars.outdir / "K_tau";
	if (fwrite_dense_matrix(cov_mat, output_path) == EXIT_SUCCESS) {
		cout << "Equal-time covariance matrix K(tau) written into file \"" << output_path.string() << "\"\n";
	}
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";
}

const int import_spks(vector<vector<int>> &spks, path &outfile_spks)
{
	vector<int> row_buf;
	string line, elem;
	ifstream ifs(outfile_spks);
	if (ifs.is_open()) {
		while(getline(ifs, line, '\n')) {
			stringstream ss(line);
			while(getline(ss, elem, '\t')) { row_buf.push_back(stof(elem)); }
			row_buf.erase(row_buf.begin()); // delete the number of spikes
			spks.push_back(row_buf);
			row_buf.clear();
		}
		ifs.close();
	} else { cout << "error in opening file."; }
	return EXIT_SUCCESS;
}

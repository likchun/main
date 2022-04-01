/**
 * @file find_cov_mat_spks.cpp
 * @author likchun@outlook.com
 * @brief find the equal-time K(0) and time-lagged K(tau) covariance matrices from
 * 		  spike trains data file "spks.txt" generated from start_simulation
 * @version 0.1
 * @date 2022-04-01
 * 
 * @copyright free to use
 * 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

struct Vars
{
	/* Settings */
    int		N			= 4095;
    double	tau			= 0.1;
    double	Tn			= 10000;
    double	dt			= 0.05;

	bool	findK0		= true;
	bool	findKT		= true;

    string	infile		= "spks.txt";
    string	outf_K0		= "K_0";
    string	outf_Ktau	= "K_tau";
} vars;

class Chronometer
{
public:
	Chronometer () {
		isActivated = false;
	}
	// Start the stopwatch, get elapsed time using stopwatch_elapsed_t() or stopwatch_lap()
	void stopwatch_begin() {
		isActivated = true;
		beg = chrono::steady_clock::now();
		lap_beg = beg;
	}
	// Return the time elapsed in millisecond (ms) since the last time calling Chronometer.stopwatch_lap(),
	// then start a new lap. Return -1 if Chronometer.stopwatch_start() has not been called
	double stopwatch_lap() {
		if (isActivated) {
			lap_last = lap_beg;
			lap_beg = chrono::steady_clock::now();
			return (double)chrono::duration_cast<chrono::milliseconds>(lap_beg - lap_last).count();
		} else {
			return -1;
		}
	}
	void destroy() { this->~Chronometer(); }
	~Chronometer() {}
private:
	chrono::steady_clock::time_point beg, lap_beg, lap_last;
	bool isActivated;
} timer;

int import_spks(vector<vector<int>>&, string&);
int fwrite_dense_matrix(vector<vector<double>>&, string);

int main()
{
	timer.stopwatch_begin();

	int lag_s = (int)(vars.tau/vars.dt);
	double sum_s_ij, total_step = (int)(vars.Tn/vars.dt);
	vector<vector<double>> cov_mat(vars.N, vector<double>(vars.N));
    vector<vector<int>> spks, spks_lag; // Record the time step of spikes
    vector<double> avg_s(vars.N);

    if (import_spks(spks, vars.infile) == EXIT_FAILURE) { return EXIT_FAILURE; }
	cout << "Spike trains data read from text file \"" << vars.infile << "\"\n";
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";

	// Calculate equal-time covariance matrix K_ij(0)
	cout << "Finding equal-time covariance matrix K(0)\n";
	if (vars.findK0)
	{
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			avg_s[i] = (double)spks[i].size() / total_step;
		}
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j <= i; j++) {
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
				cov_mat[i][j] = sum_s_ij/total_step - avg_s[i]*avg_s[j];
			}
		}
		// K(0) is symmetric
		#pragma omp parallel for
		for (int j = 0; j < vars.N; j++) {
			for (int i = 0; i < j; i++) {
				cov_mat[i][j] = cov_mat[j][i];
			}
		}
		if (fwrite_dense_matrix(cov_mat, vars.outf_K0) == EXIT_SUCCESS) {
			cout << "K(0) is written into file \"" << vars.outf_K0 << "\"\n";
		} else { return EXIT_FAILURE; }
		vector<vector<double>>(vars.N, vector<double>(vars.N, 0)).swap(cov_mat);
		cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";
	}

	// Calculate time-lagged covariance matrix K_ij(tau)
	cout << "Finding time-lagged covariance matrix K(tau)\n";
	if (vars.findKT)
	{
		spks_lag = spks;
		for (int i = 0; i < vars.N; i++) {
			for (auto &s : spks_lag[i]) {
				s += lag_s;
			}
		}
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j < vars.N; j++) {
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
				cov_mat[i][j] = sum_s_ij/total_step - avg_s[i]*avg_s[j];
			}
		}
		if (fwrite_dense_matrix(cov_mat, vars.outf_Ktau) == EXIT_SUCCESS) {
			cout << "K(tau) is written into file \"" << vars.outf_Ktau << "\"\n";
		} else { return EXIT_FAILURE; }
		cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n" << endl;
	}

	return EXIT_SUCCESS;
}

int import_spks(vector<vector<int>> &spks, string &infile_spks)
{
	spks = vector<vector<int>>();
	vector<int> row_buf;
	string line, elem;
	ifstream ifs(infile_spks);
	if (ifs.is_open()) {
		while(getline(ifs, line, '\n')) {
			stringstream ss(line);
			while(getline(ss, elem, '\t')) { row_buf.push_back(stof(elem)); }
			row_buf.erase(row_buf.begin()); // delete the number of spikes
			spks.push_back(row_buf);
			row_buf.clear();
		}
		ifs.close();
	} else {
		cout << "FileAccessError: spike train data file \""
			 << infile_spks << "\" cannot be accessed\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int fwrite_dense_matrix(vector<vector<double>> &matrix, string filename)
{
	ofstream ofs(filename, ios::out);
	if (ofs.is_open()) {
		ofs << matrix[0][0];
		for (size_t j = 1; j < matrix[0].size(); j++) {
			ofs << '\t' << matrix[0][j];
		}
		for (size_t i = 1; i < matrix.size(); i++) {
			ofs << '\n';
			ofs << matrix[i][0];
			for (size_t j = 1; j < matrix[i].size(); j++) {
				ofs << '\t' << matrix[i][j];
			}
		}
		ofs.close();
	} else {
		cout << "FileAccessError: \"" << filename << "\" cannot be written into";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

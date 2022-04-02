/**
 * @file find_cov_mat_memp.cpp
 * @author likchun@outlook.com
 * @brief find the equal-time K(0) and time-lagged K(tau) covariance matrices from
 * 		  potential time series binary file "memp.dat" generated from start_simulation
 * @version 0.3
 * @date 2022-04-01
 * 
 * @copyright free to use
 * 
 */

#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

struct Vars
{
    /* Settings */
    int		N		= 4095;
    double	tau		= 0.1;
    double	dt		= 0.05;

	bool	findK0	= true;
	bool	findKT	= true;

    string	infile	= "memp.dat";
    // string	infile	= "spk_bin_signal.dat";
    string	outf_K0	= "K_0";
    string	outf_KT	= "K_tau";
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

int seek_from_binary(vector<vector<float>>&, string, int, int);
int seek_from_binary(vector<vector<signed char>>&, string, int, int);
int fwrite_dense_matrix(vector<vector<double>>&, string);

int main()
{
	timer.stopwatch_begin();

	int lag_t = (int)(vars.tau/vars.dt);
	vector<vector<float>> memp;
	// vector<vector<signed char>> memp;
	vector<vector<double>> cov_mat(vars.N, vector<double>(vars.N, 0));
	vector<float> t_avg_v(vars.N);

	cout << "Reading potential time series\n";
	if (seek_from_binary(memp, vars.infile, 0, vars.N) == EXIT_FAILURE) { return EXIT_FAILURE; }
	cout << "Potential time series data read from binary file \"" << vars.infile << "\"\n";
	cout << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";

	// Calculate equal-time covariance matrix K_ij(0)
	if (vars.findK0)
	{
		cout << "Finding equal-time covariance matrix K(0)\n";
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			t_avg_v[i] = accumulate(memp[i].begin(), memp[i].end(), 0.0) / memp[i].size();
		}
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j <= i; j++) {
				cov_mat[i][j] = inner_product(
					memp[i].begin(), memp[i].end(), memp[j].begin(), 0.0
				) / memp[i].size() - t_avg_v[i]*t_avg_v[j];
			}
		}
		// K(0) is symmetric
		#pragma omp parallel for
		for (int j = 0; j < vars.N; j++) {
			for (int i = 0; i < j; i++) {
				cov_mat[i][j] = cov_mat[j][i];
			}
		}
		cout << "K(0) found in " << (int)(timer.stopwatch_lap()/1000) << " s\n";
		if (fwrite_dense_matrix(cov_mat, vars.outf_K0) == EXIT_SUCCESS) {
			cout << "K(0) written into file \"" << vars.outf_K0 << "\"\n\n";
		}
	}
	vector<vector<double>>(vars.N, vector<double>(vars.N, 0)).swap(cov_mat);

	// Calculate time-lagged covariance matrix K_ij(tau)
	if (vars.findKT)
	{
		cout << "Finding time-lagged covariance matrix K(tau), with tau=" << vars.tau << "\n";
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j < vars.N; j++) {
				cov_mat[i][j] = inner_product(
					memp[i].begin()+lag_t, memp[i].end(), memp[j].begin(), 0.0
				) / (memp[i].size()-lag_t) - t_avg_v[i]*t_avg_v[j];
			}
		}
		cout << "K(tau) found in " << (int)(timer.stopwatch_lap()/1000) << " s\n";
		if (fwrite_dense_matrix(cov_mat, vars.outf_KT) == 0) {
			cout << "K(tau) written into file \"" << vars.outf_KT << "\"\n\n";
		}
	}

	return EXIT_SUCCESS;
}

int seek_from_binary(vector<vector<float>> &memp, string filename, int start_idx, int mat_size)
{
	memp = vector<vector<float>>(mat_size, vector<float>());
	ifstream ifs(filename, ios::in | ios::binary);
	if (ifs.is_open()) {
		float content_buf;
		ifs.seekg(start_idx*sizeof(float), ios::beg);
		for (size_t i = 0; i < memp.size(); i++) {
			ifs.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
			memp[i].push_back(content_buf);
		}
		while (ifs.eof() == 0) {
			ifs.seekg((mat_size-memp.size())*sizeof(float), ios::cur);
			for (size_t i = 0; i < memp.size(); i++) {
				ifs.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
				memp[i].push_back(content_buf);
			}
		}
		for (size_t i = 0; i < memp.size(); i++) {
			memp[i].pop_back();
		}
		ifs.close();
	} else {
		cout << "FileAccessError: potential time series data file \""
			 << filename << "\" cannot be accessed\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int seek_from_binary(vector<vector<signed char>> &memp, string filename, int start_idx, int mat_size)
{
	memp = vector<vector<signed char>>(mat_size, vector<signed char>());
	ifstream ifs(filename, ios::in | ios::binary);
	if (ifs.is_open()) {
		signed char content_buf;
		ifs.seekg(start_idx*sizeof(signed char), ios::beg);
		for (size_t i = 0; i < memp.size(); i++) {
			ifs.read(reinterpret_cast<char*>(&content_buf), sizeof(signed char));
			memp[i].push_back(content_buf);
		}
		while (ifs.eof() == 0) {
			ifs.seekg((mat_size-memp.size())*sizeof(signed char), ios::cur);
			for (size_t i = 0; i < memp.size(); i++) {
				ifs.read(reinterpret_cast<char*>(&content_buf), sizeof(signed char));
				memp[i].push_back(content_buf);
			}
		}
		for (size_t i = 0; i < memp.size(); i++) {
			memp[i].pop_back();
		}
		ifs.close();
	} else {
		cout << "FileAccessError: potential time series data file \""
			 << filename << "\" cannot be accessed\n";
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

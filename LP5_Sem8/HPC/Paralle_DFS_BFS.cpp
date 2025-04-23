#include <iostream>
#include <omp.h>
#include <stack>
#include <queue>
#include <chrono>

using namespace std;
using namespace std::chrono;

void DFS(vector<vector<int>>& adj_matrix, int n,int start) {
    vector<bool> visited(n, false);
    stack<int> s;

    visited[start] = true;
    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();
        cout << node << " ";

        for (auto x : adj_matrix[node]) {
            if (!visited[x]) {
                s.push(x);
                visited[x] = true;
            }
        }
    }
}

void BFS(vector<vector<int>>& adj, int n,int start) {
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";
        for (auto x : adj[node]) {
            if (!visited[x]) {
                q.push(x);
                visited[x] = true;
            }
        }
    }
}

void parallel_BFS(vector<vector<int>>& adj, int n,int start) {
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int size = q.size();
        vector<int> level_nodes;
        
        #pragma omp parallel 
        {
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < size; ++i) {
                int node;
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                }
                
                #pragma omp critical
                cout << node << " ";

                for (auto x : adj[node]) {
                    if (!visited[x]) {
                        #pragma omp critical
                        {
                            q.push(x);
                            visited[x] = true;
                        }
                    }
                }
            }
        }
    }
}

void parallel_DFS(vector<vector<int>>& adj_matrix, int n,int start) {
    vector<bool> visited(n, false);
    stack<int> s;

    visited[start] = true;
    s.push(start);
    
    #pragma omp parallel
    {
        while (!s.empty()) {
            int node;
            #pragma omp critical
            {
                if (!s.empty()) {
                    node = s.top();
                    s.pop();
                }
            }
            
            #pragma omp critical
            cout << node << " ";

            #pragma omp for schedule(dynamic)
            for (auto x : adj_matrix[node]) {
                if (!visited[x]) {
                    #pragma omp critical
                    {
                        s.push(x);
                        visited[x] = true;
                    }
                }
            }
        }
    }
}

void add(vector<vector<int>>& adj_matrix, int u, int v) {
    adj_matrix[u].push_back(v);
    adj_matrix[v].push_back(u);
}

int main() {
    int n;

    cout<<"Enter number of nodes : \n";
    cin>>n;
    vector<vector<int>> adj_matrix(n);

    //randomly generating the graph 
    for(int i=0;i<n;i++){
        add(adj_matrix,rand()%n,rand()%n);
    }
    
    int start=adj_matrix[0][0];
     
    cout << "\nSequential BFS : \n";
    auto start_bfs = high_resolution_clock::now();
    BFS(adj_matrix,n,start);
    auto end_bfs = high_resolution_clock::now();
    double seq_bfs = duration<double, milli>(end_bfs - start_bfs).count();
    cout << "\nTime Taken : " << seq_bfs << " ms\n\n";

    cout << "\nSequential DFS : \n";
    auto dfs_start = high_resolution_clock::now();
    DFS(adj_matrix, n,start);
    auto dfs_end = high_resolution_clock::now();
    double seq_dfs = duration<double, milli>(dfs_end - dfs_start).count();
    cout << "\nTime Taken : " << seq_dfs << " ms\n\n";


    cout << "\nParallel BFS : \n";
    auto start_parallel_bfs = high_resolution_clock::now();
    parallel_BFS(adj_matrix, n,start);
    auto end_parallel_bfs = high_resolution_clock::now();
    double par_bfs = duration<double, milli>(end_parallel_bfs - start_parallel_bfs).count();
    cout << "\nTime Taken : " << par_bfs << " ms";
    cout << "\nSpeedup Factor : " << seq_bfs / par_bfs << "\n\n";


    cout << "\nParallel DFS : \n";
    auto start_parallel_dfs = high_resolution_clock::now();
    parallel_DFS(adj_matrix, n,start);
    auto end_parallel_dfs = high_resolution_clock::now();
    double par_dfs = duration<double, milli>(end_parallel_dfs - start_parallel_dfs).count();
    cout << "\nTime Taken : " << par_dfs << " ms";
    cout << "\nSpeedup Factor : " << seq_dfs / par_dfs << "\n\n";

    return 0;
}

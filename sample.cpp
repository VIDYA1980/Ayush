#include <iostream>
#include <vector>

using namespace std;

int main()
{
    int K, S;
    cin >> K >> S;

    const int MAX_K = 25;
    const int MAX_S = 10000;

    vector<vector<vector<long long>>> dp(4, vector<vector<long long>>(MAX_S + 1, vector<long long>(MAX_K + 1, 0)));

    for (int k = 0; k <= K; k++)
    {
        dp[0][0][k] = 1;
    }

    for (int i = 1; i <= 3; i++)
    {
        for (int j = 0; j <= S; j++)
        {
            for (int k = 0; k <= K; k++)
            {
                for (int x = 0; x <= min(j, k); x++)
                {
                    dp[i][j][k] += dp[i - 1][j - x][k];
                }
            }
        }
    }

    cout << dp[3][S][K] << endl;

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;


//Sigmoide: usada para ativacao dos neuronios
double sigmoide(double x) {
    return 1 / (1 + exp(-x));
}

//Derivada da sigmoide: usada para calcular o gradiente na retropropagação
double derivadaSigmoide(double x) {
    return x * (1 - x);
}

class RedeNeural {
private:
    double peso1, peso2, vies;
    double taxaAprendizado;

public:
    RedeNeural() {
        srand(time(0));
        peso1 = ((double)rand() / RAND_MAX) * 2 - 1;
        peso2 = ((double)rand() / RAND_MAX) * 2 - 1;
        vies = ((double)rand() / RAND_MAX) * 2 - 1;
        taxaAprendizado = 0.1;
        
        cout << "Pesos e vies iniciais: " << endl;
        cout << "Peso1: " << peso1 << ", Peso2: " << peso2 << ", Viés: " << vies << endl;
    }

    void treinar(vector<vector<double>> entradas, vector<double> saidas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (size_t i = 0; i < entradas.size(); i++) {
                //Propagacao para frente
                double soma = entradas[i][0] * peso1 + entradas[i][1] * peso2 + vies;
                double saida = sigmoide(soma);

                //Calculo do erro
                double erro = saidas[i] - saida;

                //Backpropagation
                double delta = erro * derivadaSigmoide(saida);

                //ajuste dos pesos e vies
                peso1 += taxaAprendizado * delta * entradas[i][0];
                peso2 += taxaAprendizado * delta * entradas[i][1];
                vies += taxaAprendizado * delta;

                //Imprime o progresso acada 10000 epoca
                if (epoca % 10000 == 0) {
                    cout << "Época: " << epoca << ", Erro: " << erro << endl;
                    cout << "Peso1: " << peso1 << ", Peso2: " << peso2 << ", Viés: " << vies << endl;
                }
            }
        }
    }

    double prever(double entrada1, double entrada2) {
        double soma = entrada1 * peso1 + entrada2 * peso2 + vies;
        return sigmoide(soma);
    }

    string determinarLinguagem(double entrada1, double entrada2) {
        double resultado = prever(entrada1, entrada2);
        cout << "\033[1;33mResultado da previsao: " << resultado << " (C++ = 1, outra linguagem = 0)\033[0m" << endl;
        
        if (resultado >= 0.5) {
            return "\033[34mC++ é a melhor linguagem\033[0m\n";
        } else {
            return "\033[31mA outra linguagem é melhor\033[0m\n";
        }
    }
};

int main() {
    RedeNeural rn;

    vector<vector<double>> dadosTreinamento = {
        {1, 0},  //C++ vs python
        {1, 0},  //C++ vs java
        {0, 1},  //Python vs java
        {1, 0},  //C++ vs rust
        {0, 1}   //Java vs go
    };
    
    //C++ é 1
    vector<double> respostasEsperadas = {1, 1, 0, 1, 0};

    rn.treinar(dadosTreinamento, respostasEsperadas, 1000000);

    cout << "\033[1;33mC++ vs Python: \033[0m" << rn.determinarLinguagem(1, 0) << endl;
    cout << "\033[1;33mC++ vs Java: \033[0m" << rn.determinarLinguagem(1, 0) << endl;
    cout << "\033[1;33mPython vs Java: \033[0m" << rn.determinarLinguagem(0, 1) << endl;
    cout << "\033[1;33mC++ vs Rust: \033[0m" << rn.determinarLinguagem(1, 0) << endl;
    cout << "\033[1;33mJava vs Go: \033[0m" << rn.determinarLinguagem(0, 1) << endl;

    return 0;
}

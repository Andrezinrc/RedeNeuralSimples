#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;


double sigmoide(double x) {
    return 1 / (1 + exp(-x));
}
double derivadaSigmoide(double x) {
    return x * (1 - x);
}

class RedeNeural {
private:
    double peso1, peso2, pesoOculto1, pesoOculto2, vies, viesOculto;
    double taxaAprendizado;

public:
    RedeNeural() {
        srand(time(0));
        peso1 = ((double)rand() / RAND_MAX) * 2 - 1;
        peso2 = ((double)rand() / RAND_MAX) * 2 - 1;
        pesoOculto1 = ((double)rand() / RAND_MAX) * 2 - 1;
        pesoOculto2 = ((double)rand() / RAND_MAX) * 2 - 1;
        vies = ((double)rand() / RAND_MAX) * 2 - 1;
        viesOculto = ((double)rand() / RAND_MAX) * 2 - 1;
        taxaAprendizado = 0.1;
        
        cout << "Pesos e vies iniciais: " << endl;
        cout << "Peso1: " << peso1 << ", Peso2: " << peso2 << ", PesoOculto1: " << pesoOculto1 
             << ", PesoOculto2: " << pesoOculto2 << ", Viés: " << vies << ", Viés Oculto: " << viesOculto << endl;
    }

    void treinar(vector<vector<double>> entradas, vector<double> saidas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (size_t i = 0; i < entradas.size(); i++) {
                //Propagacao para frente
                double somaEntrada = entradas[i][0] * peso1 + entradas[i][1] * peso2 + vies;
                double saidaEntrada = sigmoide(somaEntrada);

                double somaOculta = saidaEntrada * pesoOculto1 + entradas[i][1] * pesoOculto2 + viesOculto;
                double saidaOculta = sigmoide(somaOculta);

                //Calculo do erro
                double erro = saidas[i] - saidaOculta;

                //Backpropagation
                double deltaOculto = erro * derivadaSigmoide(saidaOculta);
                double deltaEntrada = deltaOculto * derivadaSigmoide(saidaEntrada);

                //ajuste dos pesos e vies
                pesoOculto1 += taxaAprendizado * deltaOculto * saidaEntrada;
                pesoOculto2 += taxaAprendizado * deltaOculto * entradas[i][1];
                viesOculto += taxaAprendizado * deltaOculto;

                peso1 += taxaAprendizado * deltaEntrada * entradas[i][0];
                peso2 += taxaAprendizado * deltaEntrada * entradas[i][1];
                vies += taxaAprendizado * deltaEntrada;

                //Imprime o progresso a cada 10000 epoca
                if (epoca % 10000 == 0) {
                    cout << "Época: " << epoca << ", Erro: " << erro << endl;
                    cout << "Peso1: " << peso1 << ", Peso2: " << peso2 << ", PesoOculto1: " << pesoOculto1 
                         << ", PesoOculto2: " << pesoOculto2 << ", Viés: " << vies << ", Viés Oculto: " << viesOculto << endl;
                }
            }
        }
    }

    double prever(double entrada1, double entrada2) {
        double somaEntrada = entrada1 * peso1 + entrada2 * peso2 + vies;
        double saidaEntrada = sigmoide(somaEntrada);

        double somaOculta = saidaEntrada * pesoOculto1 + entrada2 * pesoOculto2 + viesOculto;
        return sigmoide(somaOculta);
    }

    string determinarLinguagem(double entrada1, double entrada2) {
        double resultado = prever(entrada1, entrada2);
        cout << resultado << endl;
        
        if (resultado >= 0.5) {
            return "\033[34mC++ venceu\033[0m\n";
        } else {
            return "\033[31mA outra linguagem venceu\033[0m\n";
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

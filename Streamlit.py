import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from decimal import Decimal, getcontext

# Définir la précision
getcontext().prec = 28

# Importer les données
@st.cache_data
def load_data():
    d = pd.read_csv('portfolio_data.csv')
    return pd.DataFrame(d)

returns= load_data()

# Calculer la moyenne des rendements et la matrice de covariance
mean_returns = returns[['AMZN','DPZ','BTC','NFLX']].mean()
cov_matrix = returns[['AMZN','DPZ','BTC','NFLX']].cov()


assets = ['AMZN','DPZ','BTC','NFLX']

def evaluate(portfolio, lambda_):
    portfolio = np.array(portfolio)
    portfolio_return = np.sum(mean_returns * portfolio)
    portfolio_volatility = np.sqrt(np.dot(portfolio.T, np.dot(cov_matrix, portfolio)))
    return portfolio_volatility - lambda_ * portfolio_return  # fonction d'évaluation basée sur le modèle moyenne-variance

# Algorithme génétique
def genetic_algorithm(lambda_, mutation_prob, mean_returns,crossover_prob, cov_matrix, num_assets, pop_size):
    population = []
    for _ in range(pop_size):
        portfolio = [float(Decimal(random.random())) for _ in range(num_assets)]
        portfolio_sum = sum(portfolio)
        portfolio = [round(weight/portfolio_sum,8) for weight in portfolio]  # normaliser les poids pour qu'ils somment à 1
        population.append(portfolio)

    # Initialiser le meilleur score
    best_score = float('inf')

    # Boucle principale de l'algorithme génétique
    while True:
        population.sort(key=lambda x: evaluate(x, lambda_))  # trier la population en fonction de la performance
        population = population[:pop_size//2]  # garder seulement la moitié supérieure

        # Vérifier si la performance du meilleur portefeuille s'est améliorée
        current_score = evaluate(population[0], lambda_)
        if abs(current_score - best_score) < 1e-6:  # critère d'arrêt : la performance ne s'améliore plus
            break
        best_score = current_score

        # Générer la prochaine génération par croisement et mutation
        for _ in range(pop_size - len(population)):
            if random.random() < 0.8:  # 80% de chance de croisement
                # Sélectionner deux parents au hasard
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                # Effectuer un croisement à trois points
                crossover_points = sorted(random.sample(range(num_assets), 3))
                child = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:crossover_points[2]] + parent2[crossover_points[2]:]
                # Normaliser les poids du portefeuille de l'enfant pour qu'ils somment à 1
                child_sum = sum(child)
                child = [round(weight0/child_sum,8) for weight0 in child]
                population.append(child)
            else:  # mutation
                if random.random() < mutation_prob:  # probabilité de mutation
                    # Sélectionner un parent au hasard
                    parent = random.choice(population)
                    # Effectuer une mutation en changeant un poids au hasard
                    mutation_point = random.randint(0, num_assets-1)
                    parent[mutation_point] = random.random()
                    # Normaliser les poids du portefeuille du parent pour qu'ils somment à 1
                    parent_sum = sum(parent)
                    parent = [round(weight1/parent_sum,8) for weight1 in parent]
                    population.append(parent)
            
        # print(population)

    # Retourner le meilleur portefeuille trouvé
    best_portfolio = min(population, key=lambda x: evaluate(x, lambda_))
    return best_portfolio, best_score

#Interface utilisateur
def main():
    st.title("Optimisation de portefeuille avec un algorithme génétique")

    # Charger les données
    data = load_data()

    # Calculer la moyenne des rendements et la matrice de covariance
    mean_returns = data[['AMZN','DPZ','BTC','NFLX']].mean()
    cov_matrix = data[['AMZN','DPZ','BTC','NFLX']].cov()

    # Initialiser la population
    num_assets = len(mean_returns)
    pop_size = 1000  # taille de la population

    # Paramètres de l'utilisateur

    capital = st.sidebar.number_input("Capital", min_value=0.0, value=10000.0, step=100.0)  # Ajout de l'entrée du capital


    # Exécuter l'algorithme génétique
    if st.button("Exécuter l'algorithme génétique"):
        best_portfolio, best_score = genetic_algorithm(0.6, 0.07, mean_returns, cov_matrix, num_assets, 1000)
        st.write(f"Meilleur portefeuille: {best_portfolio}")
        st.write(f"Meilleur score: {best_score}")
        st.write(f"Répartition du capital: {np.array(best_portfolio) * capital}")  # Affichage de la répartition du capital
    # Créer un DataFrame pour les résultats
        results = pd.DataFrame({
            'Actif': assets,
            'Poids': best_portfolio,
            'Capital': np.array(best_portfolio) * capital
            })
                    # Afficher les résultats dans un tableau
        st.table(results)
                    # Exécuter l'algorithme génétique pour différentes valeurs de lambda
        lambda_values = np.linspace(0, 0.7, 10)  # par exemple, de 0 à 1 avec un pas de 0.01
        portfolios = []
        for lambda_ in lambda_values:
            portfolio, _ = genetic_algorithm(lambda_, 0.07, mean_returns, cov_matrix, num_assets, pop_size)
            portfolios.append(portfolio)

        # Calculer le rendement et la volatilité pour chaque portefeuille
        returns = [np.sum(mean_returns * np.array(portfolio)) for portfolio in portfolios]
        volatilities = [np.sqrt(np.dot(np.array(portfolio).T, np.dot(cov_matrix, np.array(portfolio)))) for portfolio in portfolios]

        # Tracer la frontière efficiente
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=returns / np.array(volatilities), marker='o')
        plt.grid(True)
        plt.xlabel('Volatilité attendue')
        plt.ylabel('Rendement attendu')
        plt.colorbar(label='Ratio de Sharpe')
        st.pyplot(plt.gcf())  # Afficher le graphique dans Streamlit

        # Calculer le rendement et la volatilité pour chaque portefeuille
        returns = [np.sum(mean_returns * np.array(portfolio)) for portfolio in portfolios]
        volatilities = [np.sqrt(np.dot(np.array(portfolio).T, np.dot(cov_matrix, np.array(portfolio)))) for portfolio in portfolios]



        

if __name__ == "__main__":
    main()

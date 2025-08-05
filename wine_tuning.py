import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import os

'''Esse arquivo foi criado para otimizar o modelo de classificação de vinhos, utilizando técnicas de tuning de hiperparâmetros com RandomizedSearchCV. 
   Contudo, as ordens de execução e as funções foram mantidas para facilitar a integração com o modelo original.'''

# Importar classe original
from class_desafio import WineQualityModel

class WineQualityModelTuned(WineQualityModel):
    """Extensão da classe original com funcionalidades de tuning de hiperparâmetros"""
    
    def __init__(self, file_path=None, random_state=42, quality_threshold=6):
        super().__init__(file_path, random_state, quality_threshold)
        self.best_params = None
        self.best_score = None
        self.cv_results = None

    def tune_hyperparameters(self, n_iter=50, cv_folds=5):
        """
        Otimiza hiperparâmetros usando RandomizedSearchCV
        Args:
            n_iter: Número de iterações da busca aleatória
            cv_folds: Número de folds para validação cruzada
        """
        print(f"\n{'='*60}")
        print("TUNING DE HIPERPARÂMETROS")
        print(f"{'='*60}")
        print(f"Iterações: {n_iter} | CV Folds: {cv_folds}")
        
        # Definir espaço de busca dos hiperparâmetros
        param_distributions = {
            'n_estimators': randint(100, 500),           # Número de árvores
            'max_depth': [None] + list(range(10, 50, 5)), # Profundidade máxima (confesso que eu dificilmente pensaria nisso sozinho [com meu nível de conhecimento naquele momento], mas um youtuber gringo chamado 'Code with Josh' me ajudou MUITO nessa parte)
            'min_samples_split': randint(2, 20),         # Mínimo para dividir nó
            'min_samples_leaf': randint(1, 10),          # Mínimo em folha
            'max_features': ['sqrt', 'log2', None],      # Features consideradas por divisão
            'bootstrap': [True, False],                  # Amostragem com/sem reposição
            'class_weight': [None, 'balanced', 'balanced_subsample']  # Balanceamento de classes
        }
        
        # Modelo base para busca
        base_model = RandomForestClassifier(random_state=self.random_state)
        
        # Configurar validação cruzada estratificada
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Configurar busca aleatória
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring='f1',  # F1-score é boa métrica para classes desbalanceadas
            n_jobs=-1,     # Usar todos os cores disponíveis
            verbose=1,     # Mostrar progresso
            random_state=self.random_state,
            return_train_score=True  # Para análise de overfitting
        )
        
        print("Executando busca por hiperparâmetros...")
        
        # Executar busca
        random_search.fit(self.X_train, self.y_train)
        
        # Salvar resultados
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = random_search.cv_results_
        
        # Mostrar resultados
        print(f"\n✅ Tuning concluído!")
        print(f"Melhor F1-score (CV): {self.best_score:.4f}")
        print(f"Melhores parâmetros:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return random_search.best_estimator_

    def get_tuning_summary(self):
        """
        Retorna resumo do processo de tuning
        """
        if self.best_params is None:
            return "Nenhum tuning executado ainda"
        
        summary = {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'cv_results_available': self.cv_results is not None
        }
        
        return summary

# EXEMPLO DE USO DO TUNING DE HIPERPARÂMETROS

if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("TUNING DE HIPERPARÂMETROS - RANDOM FOREST")
    print("=" * 60)
    
    # 1. CARREGAR DADOS
    print("\n1. Carregando datasets...")
    white_data = pd.read_csv(os.path.join(base_dir, 'wine+quality', 'winequality-white.csv'), sep=';')
    white_data['type'] = 0
    
    red_data = pd.read_csv(os.path.join(base_dir, 'wine+quality', 'winequality-red.csv'), sep=';')
    red_data['type'] = 1
    
    combined_data = pd.concat([white_data, red_data], ignore_index=True)
    print(f"Dataset combinado: {len(combined_data)} amostras")
    
    # 2. PREPARAR DADOS PARA TUNING
    print("\n2. Preparando dados para tuning...")
    tuned_model = WineQualityModelTuned(quality_threshold=6)
    tuned_model.load_and_prepare_data(combined_data, use_smote=False)
    
    # 3. EXECUTAR TUNING
    print("\n3. Executando tuning de hiperparâmetros...")
    best_estimator = tuned_model.tune_hyperparameters(
        n_iter=50,  # Número de iterações
        cv_folds=5   # Folds da validação cruzada
    )
    
    # 4. MOSTRAR RESULTADOS DO TUNING
    print(f"\n{'='*60}")
    print("RESULTADOS DO TUNING")
    print(f"{'='*60}")
    
    summary = tuned_model.get_tuning_summary()
    print(f"Melhor F1-score encontrado: {summary['best_score']:.4f}")
    print(f"\nMelhores hiperparâmetros:")
    for param, value in summary['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\n Tuning concluído! Use esses parâmetros no modelo principal.")
    print(f" Para treinar o modelo, use o arquivo class_desafio.py")

# Eu passei horas aprendendo, resolvendo bug, testando e ajustando todo o código. Então confesso que me deu bastante preguiça de escrever comentários detalhados pra cada linha (rs).
# Estou bem feliz!!!
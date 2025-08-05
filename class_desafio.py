import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  # Para balanceamento de classes (Passei horas testando pra achar algo que funcionasse rs)

class WineQualityModel:
    """Classe otimizada para classificação binária de qualidade de vinhos"""
    
    def __init__(self, file_path=None, random_state=42, quality_threshold=6):
        """
        Inicializa o modelo
        Args:
            quality_threshold: Corte para definir vinhos 'bons' (>= threshold) vs 'ruins' (< threshold)
        """
        self.file_path = file_path
        self.random_state = random_state
        self.quality_threshold = quality_threshold  # Define o que é "vinho bom"
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_and_prepare_data(self, data=None, use_smote=True):
        """
        Carrega, transforma e prepara os dados em uma única função
        Converte problema multiclasse (3-9) em binário (0-1)
        """
        # 1. CARREGAR DADOS
        if data is not None:
            self.data = data  # Usar dados fornecidos (dataset combinado)
        else:
            self.data = pd.read_csv(self.file_path, sep=';')  # Carregar do arquivo
        
        # 2. TRANSFORMAR EM PROBLEMA BINÁRIO
        # Cria variável binária: 1 = bom (>= threshold), 0 = ruim (< threshold)
        self.data['quality_binary'] = (self.data['quality'] >= self.quality_threshold).astype(int)
        
        # 3. SEPARAR FEATURES E TARGET
        X = self.data.drop(['quality', 'quality_binary'], axis=1, errors='ignore')  # Features (características do vinho)
        y = self.data['quality_binary']  # Target binário (0=ruim, 1=bom)
        
        # 4. DIVIDIR EM TREINO E TESTE
        # Stratify garante que as proporções das classes sejam mantidas
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # 5. APLICAR SMOTE PARA BALANCEAMENTO (opcional)
        # SMOTE gera amostras sintéticas da classe minoritária
        if use_smote:
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                print("✅ SMOTE aplicado")
            except:
                print("❌ SMOTE falhou, usando class_weight")
                use_smote = False
        
        # 6. MOSTRAR DISTRIBUIÇÃO DOS DADOS
        print(f"Dataset: {len(self.data)} amostras | Treino: {len(self.X_train)} | Teste: {len(self.X_test)}")
        binary_dist = self.data['quality_binary'].value_counts()
        print(f"Ruim (< {self.quality_threshold}): {binary_dist[0]} | Bom (>= {self.quality_threshold}): {binary_dist[1]}")
        
        return use_smote

    def train_model(self, smote_applied=False):
        """
        Treina o modelo RandomForest com parâmetros otimizados
        """
        # Se SMOTE foi aplicado, não usar class_weight (dados já balanceados)
        # Senão, usar class_weight='balanced' para lidar com desbalanceamento
        class_weight = None if smote_applied else 'balanced'
        
        # RandomForest com parâmetros otimizados para vinhos
        self.model = RandomForestClassifier(
            n_estimators=229,        # Número de árvores (mais árvores = mais estabilidade)
            max_depth=None,           # Profundidade máxima (evita overfitting)
            min_samples_split=3,    # Mínimo de amostras para dividir um nó
            min_samples_leaf=1,     # Mínimo de amostras em uma folha
            max_features='sqrt',    # Considera raiz quadrada das features (bom para alta dimensionalidade)
            bootstrap=True,          # Amostragem com reposição
            class_weight=class_weight,  # Balanceamento automático se necessário
            random_state=self.random_state
        )
        
        # Treinar o modelo
        self.model.fit(self.X_train, self.y_train)
        print("✅ Modelo treinado")

    def evaluate_model(self):
        """
        Avalia o modelo usando métricas apropriadas para classificação binária
        """
        # Fazer predições no conjunto de teste
        predictions = self.model.predict(self.X_test)
        
        # Calcular métricas principais
        accuracy = accuracy_score(self.y_test, predictions)    # % acertos totais
        precision = precision_score(self.y_test, predictions)  # % dos "bons" preditos que são realmente bons
        recall = recall_score(self.y_test, predictions)        # % dos "bons" reais que foram identificados
        f1 = f1_score(self.y_test, predictions)               # Média harmônica entre precisão e recall
        
        # Mostrar resultados
        print(f"\n=== RESULTADOS ===")
        print(f"Acurácia: {accuracy:.1%}")
        print(f"Precisão: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        print(f"F1-Score: {f1:.1%}")
        
        return accuracy, precision, recall, f1

    def plot_results(self):
        """
        Gera visualizações essenciais para interpretar o modelo
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. IMPORTÂNCIA DAS CARACTERÍSTICAS
        # Mostra quais características são mais importantes para classificar vinhos
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(8)
        
        axes[0].barh(importance['feature'][::-1], importance['importance'][::-1])
        axes[0].set_title('Top 8 Features Importantes')
        
        # 2. MATRIZ DE CONFUSÃO
        # Mostra erros e acertos do modelo
        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], cmap='Blues',
                   xticklabels=['Ruim', 'Bom'], yticklabels=['Ruim', 'Bom'])
        axes[1].set_title('Matriz de Confusão')
        
        # 3. DISTRIBUIÇÃO DE PROBABILIDADES
        # Mostra confiança do modelo nas predições
        proba = self.model.predict_proba(self.X_test)[:, 1]  # Probabilidade de ser "bom"
        y_test_array = np.array(self.y_test)
        axes[2].hist(proba[y_test_array == 0], alpha=0.7, label='Ruim', bins=15)
        axes[2].hist(proba[y_test_array == 1], alpha=0.7, label='Bom', bins=15)
        axes[2].axvline(x=0.5, color='red', linestyle='--')  # Linha de decisão
        axes[2].set_title('Probabilidades')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

# EXECUÇÃO PRINCIPAL

if __name__ == "__main__":
    import os
    
    # Encontrar diretório dos arquivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== CLASSIFICAÇÃO BINÁRIA DE VINHOS ===")
    
    # 1. CARREGAR E COMBINAR DATASETS
    # Carregar vinho branco
    white_data = pd.read_csv(os.path.join(base_dir, 'wine+quality', 'winequality-white.csv'), sep=';')
    white_data['type'] = 0  # 0 = vinho branco
    
    # Carregar vinho tinto
    red_data = pd.read_csv(os.path.join(base_dir, 'wine+quality', 'winequality-red.csv'), sep=';')
    red_data['type'] = 1    # 1 = vinho tinto
    
    # Combinar datasets
    combined_data = pd.concat([white_data, red_data], ignore_index=True)
    
    # 2. TREINAR MODELO
    # Criar modelo com threshold 6 (vinhos >= 6 são considerados "bons")
    model = WineQualityModel(quality_threshold=6)
    
    # Carregar dados e aplicar SMOTE para balanceamento
    smote_applied = model.load_and_prepare_data(combined_data, use_smote=True)
    
    # Treinar modelo
    model.train_model(smote_applied)
    
    # 3. AVALIAR E VISUALIZAR
    # Calcular métricas de performance
    accuracy, precision, recall, f1 = model.evaluate_model()
    
    # Gerar gráficos
    model.plot_results()
    
    # 4. MOSTRAR CARACTERÍSTICAS MAIS IMPORTANTES
    # Identificar quais características do vinho são mais relevantes
    importance = pd.DataFrame({
        'feature': model.X_train.columns,
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 3 características importantes:")
    for i, (feature, imp) in enumerate(importance.head(3).values):
        print(f"  {i+1}. {feature}: {imp:.3f}")
    
    print(f"\n🎯 Modelo otimizado para identificar vinhos >= 6!")

# Todos os emojis foram importados do site https://emojiterra.com/
# Minha maior dificuldade foi entender como usar o SMOTE, pois não consegui balancear os dados de treino sem ele. Confesso que o numpy também me confunde um pouco...
# Também é a primeira vez que aplico POO em um projeto, então acredito que esteja BASTANTE confuso e bagunçado. Mas farei de tudo para aprimorar meu entendimento no assunto. :)
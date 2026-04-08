import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set matplotlib style for a cleaner dashboard look
plt.style.use('default')

class ExplainerEngine:
    def __init__(self, model, X_train):
        """
        Initializes the SHAP explainer. 
        We use a generic shap.Explainer which attempts to automatically 
        choose the best explainer (Tree, Linear, etc.) based on the model.
        """
        self.model = model
        self.X_train = X_train
        
        # Initialize the explainer
        try:
            self.explainer = shap.Explainer(self.model, self.X_train)
        except Exception as e:
            print(f"Warning: Default explainer failed, falling back to KernelExplainer. Error: {e}")
            # KernelExplainer is a safe, model-agnostic fallback (though slower)
            # We sample the background data to keep it fast for the dashboard
            background = self.X_train.sample(min(100, len(self.X_train)), random_state=42)
            self.explainer = shap.KernelExplainer(self.model.predict, background)

    def get_global_feature_importance(self, X_test, max_display=10):
     """
     Generates a SHAP bar plot showing global feature importance.
     This answers: 'Overall, which features drive the model predictions?'
     """
     X_test_sample = X_test.sample(min(200, len(X_test)), random_state=42)
     shap_values = self.explainer(X_test_sample)

     fig, ax = plt.subplots(figsize=(10, 6))

      # Use newer SHAP bar plot instead of legacy summary_plot
     shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False)

     plt.title("Global Feature Impact on Predictions", fontsize=14, pad=20)
     plt.tight_layout()

     return plt.gcf()

    def get_local_explanation(self, instance, instance_index=0):
        """
        Generates a SHAP Waterfall plot for a single individual.
        This answers: 'Why did the model make this specific decision for Person X?'
        """
        shap_values = self.explainer(instance)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Check if shap_values is an Explanation object (depends on explainer used)
        if hasattr(shap_values, 'values'):
            shap.plots.waterfall(shap_values[instance_index], show=False)
        else:
            # Fallback for Kernel/Tree explainers that return raw arrays
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[instance_index], 
                base_values=self.explainer.expected_value, 
                data=instance.iloc[instance_index]
            ), show=False)
            
        plt.title(f"Decision Breakdown for Individual #{instance_index}", fontsize=14, pad=20)
        plt.tight_layout()
        
        return fig
    
    def get_bias_feature_breakdown(self, X_test, protected_attribute):
      """
      Compares SHAP values between privileged and unprivileged groups.
      This answers: 'Which features are driving bias against this group?'
      """
      X_sample = X_test.sample(min(300, len(X_test)), random_state=42)
      shap_values = self.explainer(X_sample)

      vals = shap_values.values if hasattr(shap_values, 'values') else shap_values

        # Split by protected attribute
      privileged_idx = X_sample[X_sample[protected_attribute] == 1].index
      unprivileged_idx = X_sample[X_sample[protected_attribute] == 0].index

      priv_shap = pd.DataFrame(
            vals[X_sample.index.isin(privileged_idx)],
            columns=X_sample.columns
        ).abs().mean()

      unpriv_shap = pd.DataFrame(
            vals[X_sample.index.isin(unprivileged_idx)],
            columns=X_sample.columns
        ).abs().mean()

      comparison = pd.DataFrame({
            "Privileged": priv_shap,
            "Unprivileged": unpriv_shap
        }).sort_values("Unprivileged", ascending=False).head(10)

      fig, ax = plt.subplots(figsize=(10, 6))
      comparison.plot(kind="bar", ax=ax, color=["#1D9E75", "#E24B4A"])
      ax.set_title(f"Feature Impact: Privileged vs Unprivileged ({protected_attribute})", 
                 fontsize=14, pad=20)
      ax.set_xlabel("Features")
      ax.set_ylabel("Mean |SHAP Value|")
      ax.legend(["Privileged (1)", "Unprivileged (0)"])
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()

      return fig

if __name__ == "__main__":
    from audit import load_adult_data, preprocess_adult, train_model
    from sklearn.model_selection import train_test_split
    import os

    print("Loading data...")
    df = load_adult_data()
    df = preprocess_adult(df)

    target_col = "income"

        # Split into train and test
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
        # Pass X_train directly — already encoded above
    model, encoded_cols = train_model(X_train, y_train)
    X_test = X_test.reindex(columns=encoded_cols, fill_value=0)

    print("Initializing explainer...")
    engine = ExplainerEngine(model, X_train)

    print("Generating global feature importance...")
    fig1 = engine.get_global_feature_importance(X_test)
    fig1.savefig("assets/shap_global.png")
    print("✅ shap_global.png saved!")

    print("Generating bias breakdown...")
    fig2 = engine.get_bias_feature_breakdown(X_test, "sex")
    fig2.savefig("assets/shap_bias.png")
    print("✅ shap_bias.png saved!")
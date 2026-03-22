import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class PairwiseIREvaluator:
    def __init__(self, 
                 use_model_ability: bool = True,
                 use_rater_disc: bool = True, 
                 use_rater_bias: bool = True,
                 use_prompt_disc: bool = True,
                 use_prompt_diff: bool = True,
                 use_feasibility: bool = True):
        """
        Initialize extended pairwise IRT model with configurable parameters.
        
        Args:
            use_model_ability: Include model ability parameters (θ)
            use_rater_disc: Include rater discriminability parameters (alpha_r)
            use_rater_bias: Include rater bias parameters (beta_r)
            use_prompt_disc: Include prompt discriminability parameters (alpha_p)
            use_prompt_diff: Include prompt difficulty offset parameters (gamma_p)
            use_feasibility: Include prompt feasibility parameters (lambda_p)
            
        Model Specification:
            P(A>B) = λ_p * σ( αᵣ * αₚ * (θ_A - θ_B) - βᵣ - γₚ ) + (0.5 - 0.5 * λ_p)
            
            Where each component can be enabled/disabled independently
        """
        # Store configuration flags
        self.use_model_ability = use_model_ability
        self.use_rater_disc = use_rater_disc
        self.use_rater_bias = use_rater_bias
        self.use_prompt_disc = use_prompt_disc
        self.use_prompt_diff = use_prompt_diff
        self.use_feasibility = use_feasibility
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize state
        self.fitted = False
        self.encoders = {
            'model': LabelEncoder(),
            'rater': LabelEncoder(),
            'prompt': LabelEncoder()
        }
        
    def _validate_configuration(self):
        """Ensure configuration makes sense"""
        if not any([self.use_model_ability, 
                   self.use_rater_disc, self.use_rater_bias,
                   self.use_prompt_disc, self.use_prompt_diff,
                   self.use_feasibility]):
            raise ValueError("At least one component must be enabled")
            
        # Rater discriminator requires either ability or bias
        if self.use_rater_disc and not (self.use_model_ability or self.use_rater_bias):
            raise ValueError("Rater discriminability requires either model ability or rater bias")
            
        # Prompt discriminator requires either ability or difficulty
        if self.use_prompt_disc and not (self.use_model_ability or self.use_prompt_diff):
            raise ValueError("Prompt discriminability requires either model ability or prompt difficulty")
            
    def _preprocess_data(self, df: pd.DataFrame):
        """Encode categorical variables to numeric indices"""
        df = df.copy()
        
        all_models = pd.concat([df['model_a'], df['model_b']]).unique()

        self.encoders['model'].fit(all_models)

        df['model_a'] = self.encoders['model'].transform(df['model_a'])
        df['model_b'] = self.encoders['model'].transform(df['model_b'])

        if ('rater_id' in df) and (self.use_rater_disc or self.use_rater_bias):
            df['rater_id'] = self.encoders['rater'].fit_transform(df['rater_id'])
        elif 'rater_id' in df:
            df = df.drop(columns=['rater_id'])

        if ('prompt_id' in df) and (self.use_prompt_disc or self.use_prompt_diff or self.use_feasibility):
            df['prompt_id'] = self.encoders['prompt'].fit_transform(df['prompt_id'])
        elif 'prompt_id' in df:
            df = df.drop(columns=['prompt_id'])
            
        return df

    def fit(self, data: pd.DataFrame, num_epochs: int = 2000, lr: float = 0.05):
        """
        Train pairwise IRT model with selected parameters
        
        Args:
            data: DataFrame with columns:
                - model_a: First model in comparison
                - model_b: Second model in comparison
                - outcome: 1 if model_a wins, 0 if model_b wins
                - rater_id: (Optional) ID of human rater
                - prompt_id: (Optional) ID of evaluation prompt
            num_epochs: Training iterations
            lr: Learning rate
        """
        # Data preprocessing
        self.n_models = len(np.union1d(data['model_a'].unique(), data['model_b'].unique()))
        self.n_raters = data['rater_id'].nunique() if 'rater_id' in data and (self.use_rater_disc or self.use_rater_bias) else 0
        self.n_prompts = data['prompt_id'].nunique() if 'prompt_id' in data and (self.use_prompt_disc or self.use_prompt_diff or self.use_feasibility) else 0
        
        # Preprocess data
        encoded_df = self._preprocess_data(data)
        tensor_data = torch.tensor(encoded_df.values, dtype=torch.long)
        
        def model():
            # Default values
            theta = torch.zeros(self.n_models)
            alpha_r = torch.ones(1)
            beta_r = torch.zeros(1)
            alpha_p = torch.ones(1)
            gamma_p = torch.zeros(1)
            lambda_p = torch.ones(1)
            
            # Model abilities (if enabled)
            if self.use_model_ability:
                with pyro.plate("models", self.n_models):
                    theta = pyro.sample("theta", dist.Normal(0, 1))
            
            # Rater parameters (if applicable)
            if self.n_raters > 0:
                with pyro.plate("raters", self.n_raters):
                    # Discriminability
                    if self.use_rater_disc:
                        log_alpha_r = pyro.sample("log_alpha_r", dist.Normal(0, 0.5))
                        alpha_r = torch.exp(log_alpha_r)
                    # Bias
                    if self.use_rater_bias:
                        beta_r = pyro.sample("beta_r", dist.Normal(0, 1))
            
            # Prompt parameters (if applicable)
            if self.n_prompts > 0:
                with pyro.plate("prompts", self.n_prompts):
                    # Discriminability
                    if self.use_prompt_disc:
                        log_alpha_p = pyro.sample("log_alpha_p", dist.Normal(0, 0.5))
                        alpha_p = torch.exp(log_alpha_p)
                    # Difficulty offset
                    if self.use_prompt_diff:
                        gamma_p = pyro.sample("gamma_p", dist.Normal(0, 1))
                    # Feasibility
                    if self.use_feasibility:
                        lambda_p = pyro.sample("lambda_p", dist.Beta(1, 1))
            
            # Calculate win probabilities
            with pyro.plate("data", tensor_data.shape[0]):
                a_idx = tensor_data[:, 0]
                b_idx = tensor_data[:, 1]
                outcome = tensor_data[:, 2]
                
                # Default indices
                r_idx = torch.zeros_like(a_idx)
                p_idx = torch.zeros_like(a_idx)
                
                # Column index management
                offset = 3  # Start after model_a, model_b, outcome
                
                # Set rater index if needed
                if self.n_raters > 0:
                    r_idx = tensor_data[:, offset]
                    offset += 1
                
                # Set prompt index if needed
                if self.n_prompts > 0:
                    p_idx = tensor_data[:, offset]
                
                # Calculate ability difference if enabled
                ability_diff = torch.zeros_like(a_idx, dtype=torch.float)
                if self.use_model_ability:
                    ability_diff = theta[a_idx] - theta[b_idx]
                
                # Apply discriminability parameters
                combined_alpha = alpha_r[r_idx] * alpha_p[p_idx]
                scaled_diff = combined_alpha * ability_diff
                
                # Apply bias/difficulty
                logit = scaled_diff
                if self.use_rater_bias:
                    logit -= beta_r[r_idx]
                if self.use_prompt_diff:
                    logit -= gamma_p[p_idx]
                
                # Core probability
                p_standard = torch.sigmoid(logit)
                
                # Apply feasibility parameter if enabled
                if self.use_feasibility:
                    p_obs = lambda_p[p_idx] * p_standard + 0.5 * (1 - lambda_p[p_idx])
                else:
                    p_obs = p_standard
                
                # Sample observations
                pyro.sample("obs", dist.Bernoulli(p_obs), obs=outcome.float())
        
        # Guide function with similar conditional logic
        def guide():
            # Variational parameters for model abilities
            if self.use_model_ability:
                with pyro.plate("models", self.n_models):
                    theta_loc = pyro.param("theta_loc", torch.randn(self.n_models))
                    theta_scale = pyro.param("theta_scale", torch.ones(self.n_models), 
                                            constraint=dist.constraints.positive)
                    pyro.sample("theta", dist.Normal(theta_loc, theta_scale))
            
            # Rater parameters
            if self.n_raters > 0:
                with pyro.plate("raters", self.n_raters):
                    # Discriminability
                    if self.use_rater_disc:
                        log_alpha_r_loc = pyro.param("log_alpha_r_loc", torch.zeros(self.n_raters))
                        log_alpha_r_scale = pyro.param("log_alpha_r_scale", torch.ones(self.n_raters), 
                                                      constraint=dist.constraints.positive)
                        pyro.sample("log_alpha_r", dist.Normal(log_alpha_r_loc, log_alpha_r_scale))
                    # Bias
                    if self.use_rater_bias:
                        beta_r_loc = pyro.param("beta_r_loc", torch.randn(self.n_raters))
                        beta_r_scale = pyro.param("beta_r_scale", torch.ones(self.n_raters), 
                                                constraint=dist.constraints.positive)
                        pyro.sample("beta_r", dist.Normal(beta_r_loc, beta_r_scale))
            
            # Prompt parameters
            if self.n_prompts > 0:
                with pyro.plate("prompts", self.n_prompts):
                    # Discriminability
                    if self.use_prompt_disc:
                        log_alpha_p_loc = pyro.param("log_alpha_p_loc", torch.zeros(self.n_prompts))
                        log_alpha_p_scale = pyro.param("log_alpha_p_scale", torch.ones(self.n_prompts), 
                                                     constraint=dist.constraints.positive)
                        pyro.sample("log_alpha_p", dist.Normal(log_alpha_p_loc, log_alpha_p_scale))
                    # Difficulty
                    if self.use_prompt_diff:
                        gamma_p_loc = pyro.param("gamma_p_loc", torch.randn(self.n_prompts))
                        gamma_p_scale = pyro.param("gamma_p_scale", torch.ones(self.n_prompts), 
                                                constraint=dist.constraints.positive)
                        pyro.sample("gamma_p", dist.Normal(gamma_p_loc, gamma_p_scale))
                    # Feasibility
                    if self.use_feasibility:
                        lambda_p_alpha = pyro.param("lambda_p_alpha", torch.ones(self.n_prompts), 
                                                 constraint=dist.constraints.positive)
                        lambda_p_beta = pyro.param("lambda_p_beta", torch.ones(self.n_prompts), 
                                                constraint=dist.constraints.positive)
                        pyro.sample("lambda_p", dist.Beta(lambda_p_alpha, lambda_p_beta))
        
        # Train the model
        pyro.clear_param_store()
        optim = Adam({"lr": lr})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())
        
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step()
            losses.append(loss)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        print(f"Final loss: {losses[-1]:.4f}")
        self.fitted = True
        
        # Store parameters (only store what was used)
        self.params = {}
        if self.use_model_ability:
            self.params['theta_loc'] = pyro.param("theta_loc").detach().numpy()
            self.params['theta_scale'] = pyro.param("theta_scale").detach().numpy()
        
        # Rater parameters
        if self.n_raters > 0:
            if self.use_rater_disc:
                self.params['log_alpha_r_loc'] = pyro.param("log_alpha_r_loc").detach().numpy()
                self.params['log_alpha_r_scale'] = pyro.param("log_alpha_r_scale").detach().numpy()
                self.params['alpha_r_loc'] = np.exp(self.params['log_alpha_r_loc'])
            if self.use_rater_bias:
                self.params['beta_r_loc'] = pyro.param("beta_r_loc").detach().numpy()
                self.params['beta_r_scale'] = pyro.param("beta_r_scale").detach().numpy()
        
        # Prompt parameters
        if self.n_prompts > 0:
            if self.use_prompt_disc:
                self.params['log_alpha_p_loc'] = pyro.param("log_alpha_p_loc").detach().numpy()
                self.params['log_alpha_p_scale'] = pyro.param("log_alpha_p_scale").detach().numpy()
                self.params['alpha_p_loc'] = np.exp(self.params['log_alpha_p_loc'])
            if self.use_prompt_diff:
                self.params['gamma_p_loc'] = pyro.param("gamma_p_loc").detach().numpy()
                self.params['gamma_p_scale'] = pyro.param("gamma_p_scale").detach().numpy()
            if self.use_feasibility:
                self.params['lambda_p_alpha'] = pyro.param("lambda_p_alpha").detach().numpy()
                self.params['lambda_p_beta'] = pyro.param("lambda_p_beta").detach().numpy()
                # Estimate lambda_p as mean of Beta distribution
                self.params['lambda_p_loc'] = (self.params['lambda_p_alpha'] / 
                                             (self.params['lambda_p_alpha'] + self.params['lambda_p_beta']))
            
    def get_abilities(self) -> pd.DataFrame:
        """Return model ability estimates (if enabled)"""
        if not self.fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        if not self.use_model_ability:
            raise RuntimeError("Model abilities not enabled in configuration")
            
        models = self.encoders['model'].classes_
        return pd.DataFrame({
            'model': models,
            'ability': self.params['theta_loc'],
            'std_dev': self.params['theta_scale']
        }).sort_values('ability', ascending=False)
    
    def get_rater_parameters(self) -> pd.DataFrame:
        """Return rater parameters (if enabled)"""
        if not self.fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.n_raters == 0:
            raise RuntimeError("Rater parameters not available")
            
        raters = self.encoders['rater'].classes_
        params = {'rater': raters}
        
        if self.use_rater_disc:
            params['discriminability'] = self.params['alpha_r_loc']
        if self.use_rater_bias:
            params['bias'] = self.params['beta_r_loc']
            
        return pd.DataFrame(params)
    
    def get_prompt_parameters(self) -> pd.DataFrame:
        """Return prompt parameters (if enabled)"""
        if not self.fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.n_prompts == 0:
            raise RuntimeError("Prompt parameters not available")
            
        prompts = self.encoders['prompt'].classes_
        params = {'prompt': prompts}
        
        if self.use_prompt_disc:
            params['discriminability'] = self.params['alpha_p_loc']
        if self.use_prompt_diff:
            params['difficulty_offset'] = self.params['gamma_p_loc']
        if self.use_feasibility:
            params['feasibility'] = self.params['lambda_p_loc']
            
        return pd.DataFrame(params)
    
    def predict_win_probability(self, model_a: str, model_b: str, 
                               rater: str = None, prompt: str = None) -> float:
        """Predict probability that model_a wins over model_b"""
        if not self.fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        # Default values for all parameters
        theta_a, theta_b = 0.0, 0.0
        alpha_r, beta_r = 1.0, 0.0
        alpha_p, gamma_p = 1.0, 0.0
        lambda_p = 1.0
        
        # Get model abilities if enabled
        if self.use_model_ability:
            model_idx_map = self.encoders['model'].transform
            theta_a = self.params['theta_loc'][model_idx_map([model_a])[0]]
            theta_b = self.params['theta_loc'][model_idx_map([model_b])[0]]
        
        # Get rater parameters if applicable
        if rater and self.n_raters > 0:
            rater_idx = self.encoders['rater'].transform([rater])[0]
            if self.use_rater_disc:
                alpha_r = self.params['alpha_r_loc'][rater_idx]
            if self.use_rater_bias:
                beta_r = self.params['beta_r_loc'][rater_idx]
        
        # Get prompt parameters if applicable
        if prompt and self.n_prompts > 0:
            prompt_idx = self.encoders['prompt'].transform([prompt])[0]
            if self.use_prompt_disc:
                alpha_p = self.params['alpha_p_loc'][prompt_idx]
            if self.use_prompt_diff:
                gamma_p = self.params['gamma_p_loc'][prompt_idx]
            if self.use_feasibility:
                lambda_p = self.params['lambda_p_loc'][prompt_idx]
            
        # Calculate win probability
        ability_diff = theta_a - theta_b
        scaled_diff = ability_diff * alpha_r * alpha_p
        logit = scaled_diff - beta_r - gamma_p
        p_standard = 1 / (1 + np.exp(-logit))
        
        # Apply feasibility parameter if enabled
        if self.use_feasibility:
            p_obs = lambda_p * p_standard + 0.5 * (1 - lambda_p)
        else:
            p_obs = p_standard
        
        return p_obs

    def get_configuration_summary(self):
        """Return a summary of the active model components"""
        return {
            "model_ability": self.use_model_ability,
            "rater_discriminability": self.use_rater_disc,
            "rater_bias": self.use_rater_bias,
            "prompt_discriminability": self.use_prompt_disc,
            "prompt_difficulty": self.use_prompt_diff,
            "prompt_feasibility": self.use_feasibility
        }
    
    def explain_prediction(self, model_a: str, model_b: str, 
                          rater: str = None, prompt: str = None) -> dict:
        """Detailed explanation of prediction components"""
        prediction = self.predict_win_probability(model_a, model_b, rater, prompt)
        explanation = {
            "base_prob": prediction,
            "components": {}
        }
        
        # Calculate intermediate values
        theta_a, theta_b = 0.0, 0.0
        if self.use_model_ability:
            model_idx_map = self.encoders['model'].transform
            theta_a = self.params['theta_loc'][model_idx_map([model_a])[0]]
            theta_b = self.params['theta_loc'][model_idx_map([model_b])[0]]
        
        alpha_r, beta_r = 1.0, 0.0
        if rater and self.n_raters > 0:
            rater_idx = self.encoders['rater'].transform([rater])[0]
            if self.use_rater_disc:
                alpha_r = self.params['alpha_r_loc'][rater_idx]
            if self.use_rater_bias:
                beta_r = self.params['beta_r_loc'][rater_idx]
        
        alpha_p, gamma_p = 1.0, 0.0
        if prompt and self.n_prompts > 0:
            prompt_idx = self.encoders['prompt'].transform([prompt])[0]
            if self.use_prompt_disc:
                alpha_p = self.params['alpha_p_loc'][prompt_idx]
            if self.use_prompt_diff:
                gamma_p = self.params['gamma_p_loc'][prompt_idx]
            if self.use_feasibility:
                lambda_p = self.params['lambda_p_loc'][prompt_idx]
        
        # Construct component breakdown
        if self.use_model_ability:
            explanation["components"]["ability_difference"] = theta_a - theta_b
        
        alpha = alpha_r * alpha_p
        if self.use_rater_disc or self.use_prompt_disc:
            explanation["components"]["combined_discriminability"] = alpha
        
        scaled_diff = (theta_a - theta_b) * alpha
        if (self.use_model_ability and 
            (self.use_rater_disc or self.use_prompt_disc)):
            explanation["components"]["scaled_difference"] = scaled_diff
        
        bias_sum = beta_r + gamma_p
        if self.use_rater_bias or self.use_prompt_diff:
            explanation["components"]["bias_sum"] = bias_sum
        
        logit = scaled_diff - bias_sum
        if any([self.use_model_ability, self.use_rater_disc, 
                self.use_prompt_disc, self.use_rater_bias,
                self.use_prompt_diff]):
            explanation["components"]["logit"] = logit
            explanation["components"]["base_probability"] = 1 / (1 + np.exp(-logit))
        
        if self.use_feasibility:
            explanation["components"]["feasibility_adjustment"] = {
                "lambda": lambda_p,
                "adjusted_probability": prediction
            }
        
        return explanation
    
    def visualize_parameters(self, figsize=(18, 12)):
        """
        Create visualizations of all estimated parameters in the model
        
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 3)
        
        # Create subplots based on enabled parameters
        ax_idx = 0
        axes = []
        
        # Model abilities plot
        if self.use_model_ability:
            ax = fig.add_subplot(gs[0, 0])
            abilities = self.get_abilities()
            sns.barplot(x='ability', y='model', data=abilities, ax=ax, palette='viridis')
            ax.set_title('Model Ability Estimates')
            ax.set_xlabel('Ability (θ)')
            ax.set_ylabel('')
            axes.append(ax)
            ax_idx += 1
        
        # Rater parameters plot
        if self.n_raters > 0 and (self.use_rater_disc or self.use_rater_bias):
            ax = fig.add_subplot(gs[0, 1])
            raters = self.get_rater_parameters()
            
            # Create scatter plot with discriminability and bias
            if self.use_rater_disc and self.use_rater_bias:
                sns.scatterplot(x='bias', y='discriminability', data=raters, 
                            ax=ax, s=100, hue='discriminability', palette='coolwarm')
                ax.set_title('Rater Characteristics')
                ax.set_xlabel('Bias')
                ax.set_ylabel('Discriminability')
                ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(0.0, color='red', linestyle='--', alpha=0.5)
                
            # Plot just discriminability
            elif self.use_rater_disc:
                sns.barplot(x='discriminability', y='rater', data=raters, 
                        ax=ax, palette='viridis')
                ax.set_title('Rater Discriminability')
                ax.set_xlabel('Discriminability')
                ax.axvline(1.0, color='red', linestyle='--')
                
            # Plot just bias
            elif self.use_rater_bias:
                sns.barplot(x='bias', y='rater', data=raters, 
                        ax=ax, palette='coolwarm')
                ax.set_title('Rater Bias')
                ax.set_xlabel('Bias')
                ax.axvline(0.0, color='red', linestyle='--')
                
            axes.append(ax)
            ax_idx += 1
        
        # Prompt parameters plot
        if self.n_prompts > 0:
            ax = fig.add_subplot(gs[0, 2])
            prompts = self.get_prompt_parameters()
            
            # Create scatter plot with discriminability and difficulty
            if self.use_prompt_disc and self.use_prompt_diff:
                sns.scatterplot(x='difficulty_offset', y='discriminability', 
                            data=prompts, ax=ax, s=100, 
                            hue='feasibility' if self.use_feasibility else None,
                            palette='viridis')
                ax.set_title('Prompt Characteristics')
                ax.set_xlabel('Difficulty Offset')
                ax.set_ylabel('Discriminability')
                ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(0.0, color='red', linestyle='--', alpha=0.5)
                
            # Plot just discriminability
            elif self.use_prompt_disc:
                sns.barplot(x='discriminability', y='prompt', data=prompts, 
                        ax=ax, palette='viridis')
                ax.set_title('Prompt Discriminability')
                ax.set_xlabel('Discriminability')
                ax.axvline(1.0, color='red', linestyle='--')
                
            # Plot just difficulty
            elif self.use_prompt_diff:
                sns.barplot(x='difficulty_offset', y='prompt', data=prompts, 
                        ax=ax, palette='coolwarm')
                ax.set_title('Prompt Difficulty Offset')
                ax.set_xlabel('Difficulty Offset')
                ax.axvline(0.0, color='red', linestyle='--')
                
            axes.append(ax)
            ax_idx += 1
        
        # Feasibility histogram
        if self.use_feasibility and self.n_prompts > 0:
            ax = fig.add_subplot(gs[1, :])
            sns.histplot(self.params['lambda_p_loc'], bins=20, kde=True, 
                        ax=ax, color='skyblue')
            ax.set_title('Prompt Feasibility Distribution')
            ax.set_xlabel('Feasibility (λ)')
            ax.set_ylabel('Count')
            ax.axvline(0.5, color='red', linestyle='--')
            axes.append(ax)
            ax_idx += 1
        
        # Parameter correlations
        if self.use_model_ability and self.n_prompts > 0:
            ax = fig.add_subplot(gs[2, :])
            
            # Create a matrix of model abilities vs prompt difficulties
            abilities = self.params['theta_loc']
            prompt_params = self.get_prompt_parameters()
            
            # Create a grid of model-prompt combinations
            models = self.encoders['model'].classes_
            prompts = self.encoders['prompt'].classes_
            
            # Calculate expected win probability for each model-prompt pair
            # against an average model (ability=0)
            prob_matrix = np.zeros((len(models), len(prompts)))
            
            for i, model in enumerate(models):
                for j, prompt in enumerate(prompts):
                    ability = abilities[i]
                    difficulty = prompt_params.loc[prompt_params['prompt'] == prompt, 
                                                'difficulty_offset'].values[0]
                    discriminability = prompt_params.loc[prompt_params['prompt'] == prompt, 
                                                    'discriminability'].values[0] if self.use_prompt_disc else 1.0
                    feasibility = prompt_params.loc[prompt_params['prompt'] == prompt, 
                                                'feasibility'].values[0] if self.use_feasibility else 1.0
                    
                    # Calculate win probability against average model
                    logit = discriminability * (ability - 0) - difficulty
                    p_standard = 1 / (1 + np.exp(-logit))
                    
                    # Apply feasibility
                    if self.use_feasibility:
                        p_obs = feasibility * p_standard + 0.5 * (1 - feasibility)
                    else:
                        p_obs = p_standard
                        
                    prob_matrix[i, j] = p_obs
            
            # Create heatmap
            sns.heatmap(prob_matrix, ax=ax, cmap='viridis', 
                    xticklabels=prompts, yticklabels=models,
                    cbar_kws={'label': 'Win Probability vs Average Model'})
            ax.set_title('Model Performance by Prompt')
            ax.set_xlabel('Prompt')
            ax.set_ylabel('Model')
            axes.append(ax)
        
        # Add overall title
        fig.suptitle('Pairwise IRT Model Parameter Visualizations', fontsize=16)
        
        return fig
    
# Enhanced simulated data generator with feasibility parameters
def simulate_pairwise_data():
    # True parameters
    np.random.seed(42)
    torch.manual_seed(42)
    
    rater_num = 10
    prompt_num = 100
    model_abilities = {"GPT-4": 1.5, "Claude": .6, "Llama": -.3, "Falcon": -1.2}
    rater_biases = {f"Rater-{i}": np.random.normal(0, 0.5) for i in range(1, rater_num+1)}
    rater_disc = {f"Rater-{i}": np.exp(np.random.normal(0, 0.3)) for i in range(1, rater_num+1)}
    prompt_diffs = {f"Prompt-{i}": np.random.normal(0, 0.3) for i in range(1, prompt_num+1)}
    prompt_disc = {f"Prompt-{i}": np.exp(np.random.normal(0, 0.2)) for i in range(1, prompt_num+1)}
    prompt_feas = {f"Prompt-{i}": np.random.beta(1.0, 1.0) for i in range(1, prompt_num+1)}
    
    # Generate comparisons
    data = []
    models = list(model_abilities.keys())
    
    for _ in range(2000):  # number of pairwise test
        prompt = np.random.choice(list(prompt_diffs.keys()))
        rater = np.random.choice(list(rater_biases.keys()))
        model_a, model_b = np.random.choice(models, size=2, replace=False)
        
        # True win probability without feasibility
        ability_diff = model_abilities[model_a] - model_abilities[model_b]
        scaled_diff = ability_diff * rater_disc[rater] * prompt_disc[prompt]
        p_standard = 1 / (1 + np.exp(-(scaled_diff - rater_biases[rater] - prompt_diffs[prompt])))
        
        # Apply feasibility parameter
        true_prob = prompt_feas[prompt] * p_standard + 0.5 * (1 - prompt_feas[prompt])
        
        # Generate outcome
        outcome = np.random.binomial(1, true_prob)
        
        data.append({
            'model_a': model_a,
            'model_b': model_b,
            'outcome': outcome,
            'rater_id': rater,
            'prompt_id': prompt
        })
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with all parameters
    print("Generating simulated pairwise comparison data...")
    pairwise_data = simulate_pairwise_data()
    print(f"Generated {len(pairwise_data)} comparisons")
    
    # Initialize and train full model
    print("\nTraining extended pairwise IRT model with feasibility parameters...")
    evaluator = PairwiseIREvaluator()
    evaluator.fit(pairwise_data, num_epochs=5000, lr=0.02)
    
    # Display results
    print("\nModel Ability Estimates:")
    print(evaluator.get_abilities())
    
    print("\nRater Parameters:")
    print(evaluator.get_rater_parameters())
    
    print("\nPrompt Parameters:")
    prompt_params = evaluator.get_prompt_parameters()
    print(prompt_params[['prompt', 'feasibility', 'difficulty_offset', 'discriminability']])
    
    # Generate visualizations
    fig = evaluator.visualize_parameters()
    plt.savefig("parameter_visualizations.png", bbox_inches='tight')
    print("\nSaved parameter visualizations to parameter_visualizations.png")
    
    # Example prediction
    print("\nPrediction Example:")
    model_a, model_b = "GPT-4", "Llama"
    rater, prompt = "Rater-1", "Prompt-1"
    prob = evaluator.predict_win_probability(model_a, model_b, rater, prompt)
    print(f"P({model_a} > {model_b} | {rater}, {prompt}) = {prob:.3f}")
    
    # Feasibility report
    if evaluator.use_feasibility and evaluator.n_prompts > 0:
        print("\nPrompt Feasibility Report:")
        unfeasible = prompt_params[prompt_params['feasibility'] < 0.6]
        if len(unfeasible) > 0:
            print(f"Identified {len(unfeasible)} low-feasibility prompts:")
            print(unfeasible[['prompt', 'feasibility']])
        else:
            print("All prompts show adequate feasibility (λ >= 0.6)")
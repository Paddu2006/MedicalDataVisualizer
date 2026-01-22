import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    df = pd.read_csv('medical_examination.csv')
    return df

def add_overweight_column(df):
    height_in_m = df['height'] / 100
    df['bmi'] = df['weight'] / (height_in_m ** 2)
    df['overweight'] = (df['bmi'] > 25).astype(int)
    df.drop('bmi', axis=1, inplace=True)
    return df

def normalize_data(df):
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
    return df

def draw_cat_plot():
    df = load_data()
    df = add_overweight_column(df)
    df = normalize_data(df)
    
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
    
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')
    df_cat.rename(columns={'value': 'category'}, inplace=True)
    
    g = sns.catplot(
        x='variable',
        y='total',
        hue='category',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1.2,
        palette='Set2'
    )
    
    g.set_axis_labels("Variable", "Total Count")
    g.set_titles("Cardio Disease: {col_name}")
    g.set_xticklabels(rotation=45)
    
    fig = g.fig
    fig.savefig('assets/catplot.png')
    
    return fig

def draw_heat_map():
    df = load_data()
    df = add_overweight_column(df)
    df = normalize_data(df)
    
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Medical Examination Data Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    fig.savefig('assets/heatmap.png', dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    cat_fig = draw_cat_plot()
    heat_fig = draw_heat_map()
    print("✅ Plots generated successfully!")
    print("📊 Categorical plot saved to: assets/catplot.png")
    print("🔥 Heatmap saved to: assets/heatmap.png")

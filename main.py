import medical_data_visualizer as mdv

if __name__ == "__main__":
    print("Testing Medical Data Visualizer...")
    
    df = mdv.load_data()
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    df_with_overweight = mdv.add_overweight_column(df.copy())
    print(f"✅ Overweight column added!")
    print(f"   Overweight distribution:\n{df_with_overweight['overweight'].value_counts()}")
    
    df_normalized = mdv.normalize_data(df_with_overweight.copy())
    print(f"✅ Data normalized!")
    
    print("\nGenerating plots...")
    cat_fig = mdv.draw_cat_plot()
    heat_fig = mdv.draw_heat_map()
    
    print("\n🎉 All tests passed! Plots saved to assets/ folder.")

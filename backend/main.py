from orchestrator import Orchestrator

def main():
    orchestrator = None
    try:
        
        # Initialize Orchestrator
        orchestrator = Orchestrator()
        
        # csv_paths_input = input("请输入要分析的 CSV 文件路径 (多个文件请用逗号分隔):\n> ")
        csv_paths_input = "大模型实习项目测试.csv"
        csv_files = [path.strip() for path in csv_paths_input.split(',') if path.strip()]
        
        if not csv_files:
            print("No CSV files provided, the program will exit.")
            return
            
        # Load CSV files into the workspace
        orchestrator.load_csvs(csv_files)
        
        # Display initial dataframe summaries
        # print(orchestrator.state_manager.get_dataframe_summaries_for_display())

        while True:
            user_input = input("[user]: ")
            
            if user_input.lower() == ':q':
                break
                
            if not user_input:
                print("please input a valid question.")
                continue
            
            # 调用 Orchestrator 执行分析
            orchestrator.run_analysis(user_input)
        # orchestrator.run_analysis("分析 Clothing 随时间变化的总销售额趋势.")
            
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nAn serious error occurred: {e}")
    finally:
        # Ensure the kernel is properly shut down
        if orchestrator:
            orchestrator.shutdown()

if __name__ == "__main__":
    main()

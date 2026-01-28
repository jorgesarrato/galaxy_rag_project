from vectorize_pdf import IngestionPipeline
def main():
    print("Checking for new source files...")
    pipeline = IngestionPipeline()
    pipeline.run()
    
if __name__ == "__main__":
    main()

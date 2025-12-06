            # Search using FAISS
            results = index_manager.search(query_embedding, k=len(file_paths), threshold=threshold_normalized)

            # Apply deduplication if enabled
            if remove_duplicates:
                print(f"\n{'='*60}")
                print(f"🔍 DEDUPLICATION ENABLED")
                print(f"   Total results from FAISS: {len(results)}")
                print(f"{'='*60}\n")

                deduplicated_results, duplicates_removed, duplicates_log = deduplicate_faiss_results(
                    results,
                    file_paths,
                    duplicate_threshold=0.97
                )

                print(f"📊 Deduplication: {len(results)} results → {len(deduplicated_results)} unique ({duplicates_removed} duplicates removed)")
                for duplicate in duplicates_log[:5]:
                    print(f"   • {duplicate['path']} → {duplicate['reason']}")

                yield f"data: {json.dumps({'type': 'info', 'message': f'📊 Deduplication removed {duplicates_removed} duplicates'})}\n\n"
                
                final_results = deduplicated_results
            else:
                print(f"\n🔍 DEDUPLICATION DISABLED - Showing all {len(results)} results\n")
                final_results = results
            
            # Stream results
            for idx, similarity in final_results:
                match = {
                    'path': file_paths[int(idx)],
                    'similarity': float(similarity * 100)  # Convert back to 0-100
                }
                yield f"data: {json.dumps({'type': 'match', 'match': match})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete', 'total_processed': len(file_paths), 'matches': len(final_results)})}\n\n"

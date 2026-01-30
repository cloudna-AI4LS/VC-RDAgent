#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for rare disease diagnosis MCP server
Tests all endpoints defined in test_rare_disease_diagnosis.http
Directly calls tool functions instead of HTTP requests
"""
import sys
import os
import json
import time

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_simple_tool'))

# Set UTF-8 encoding for stdout to handle Chinese characters properly
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import tool functions
from tools.phenotype_extractor import phenotype_extractor
from tools.disease_case_extractor import extract_disease_cases
from tools.disease_information_retrieval import disease_information_retrieval
from tools.disease_diagnosis import disease_diagnosis

def safe_json_dumps(obj, indent=2):
    """Safely dump JSON with proper encoding handling"""
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        try:
            return json.dumps(obj, indent=indent, ensure_ascii=True)
        except Exception:
            return f"<Error encoding JSON: {e}>"

def print_result(test_name, result):
    """Print test result"""
    print("=" * 80)
    print(f"Test: {test_name}")
    print("=" * 80)
    
    if isinstance(result, str):
        try:
            # Try to parse as JSON for pretty printing
            parsed = json.loads(result)
            print(f"Response: {safe_json_dumps(parsed)}")
        except json.JSONDecodeError:
            print(f"Response: {result}")
    else:
        print(f"Response: {safe_json_dumps(result)}")
    
    print()

def test_phenotype_extractor_basic():
    """Test 2.1: Phenotype extractor - Basic query"""
    result = phenotype_extractor(
        query="Patient presents with intellectual disability, developmental delay, and hypotonia"
    )
    print_result("Phenotype Extractor - Basic Query", result)
    return result

def test_phenotype_extractor_with_hpo():
    """Test 2.2: Phenotype extractor - Query with HPO IDs"""
    result = phenotype_extractor(
        query="Patient has HP:0001250 and HP:0001249 symptoms, showing intellectual developmental delay"
    )
    print_result("Phenotype Extractor - Query with HPO IDs", result)
    return result

def test_phenotype_extractor_init_only():
    """Test 2.3: Phenotype extractor - Initialize only"""
    result = phenotype_extractor(
        query="",
        only_init=True
    )
    print_result("Phenotype Extractor - Initialize Only", result)
    return result

def test_phenotype_extractor_full_params():
    """Test 2.4: Phenotype extractor - Full parameters"""
    result = phenotype_extractor(
        query="Patient has seizures, intellectual disability, microcephaly",
        include_categories=True
    )
    print_result("Phenotype Extractor - Full Parameters", result)
    return result

def test_phenotype_extractor_complex():
    """Test 2.6: Phenotype extractor - Complex clinical description"""
    result = phenotype_extractor(
        query="A 5-year-old boy presents with global developmental delay, hypotonia, seizures, microcephaly, characteristic facial dysmorphism, and feeding difficulties"
    )
    print_result("Phenotype Extractor - Complex Clinical Description", result)
    return result

def test_disease_info_with_model():
    """Test 3.1: Disease information retrieval - Extract from query with model"""
    result = disease_information_retrieval(
        query="Patient may have Down syndrome or Trisomy 21",
        use_model=True
    )
    print_result("Disease Information Retrieval - With Model", result)
    return result

def test_disease_info_with_id():
    """Test 3.2: Disease information retrieval - With disease ID"""
    result = disease_information_retrieval(
        query="OMIM:190685",
        use_model=True
    )
    print_result("Disease Information Retrieval - With Disease ID", result)
    return result

def test_disease_info_with_candidates():
    """Test 3.3: Disease information retrieval - With candidates (no model)"""
    result = disease_information_retrieval(
        query="candidates provided",
        use_model=False,
        candidates=["Down syndrome", "Trisomy 21", "OMIM:190685"]
    )
    print_result("Disease Information Retrieval - With Candidates", result)
    return result

def test_disease_info_multiple():
    """Test 3.5: Disease information retrieval - Multiple disease names"""
    result = disease_information_retrieval(
        query="The patient might have Rett syndrome, Angelman syndrome, or Prader-Willi syndrome",
        use_model=True
    )
    print_result("Disease Information Retrieval - Multiple Disease Names", result)
    return result

def test_extract_disease_cases_basic():
    """Test 4.1: Extract disease cases - Basic query"""
    result = extract_disease_cases(
        query="Patient presents with intellectual disability, developmental delay, and hypotonia"
    )
    print_result("Extract Disease Cases - Basic Query", result)
    return result

def test_extract_disease_cases_custom_k():
    """Test 4.2: Extract disease cases - With custom top_k parameters"""
    result = extract_disease_cases(
        query="Patient has seizures, microcephaly, and intellectual disability",
        top_k=50,
        final_top_k=20
    )
    print_result("Extract Disease Cases - Custom top_k Parameters", result)
    return result

def test_extract_disease_cases_complex():
    """Test 4.3: Extract disease cases - Complex phenotype description"""
    result = extract_disease_cases(
        query="A child with global developmental delay, hypotonia, seizures, microcephaly, characteristic facial dysmorphism, and feeding difficulties",
        top_k=100,
        final_top_k=30
    )
    print_result("Extract Disease Cases - Complex Phenotype Description", result)
    return result

def test_extract_disease_cases_hpo_only():
    """Test 4.5: Extract disease cases - Query with HPO IDs only"""
    result = extract_disease_cases(
        query="HP:0001250 HP:0001249 HP:0001252",
        top_k=100,
        final_top_k=50
    )
    print_result("Extract Disease Cases - HPO IDs Only", result)
    return result

def test_disease_diagnosis_full():
    """Test 5.1: Disease diagnosis - Full workflow"""
    result = disease_diagnosis(
        original_query="Patient presents with intellectual disability, developmental delay, and hypotonia",
        extracted_phenotypes={
            "HP:0001250,HP:0001249,HP:0001252": {
                "Intellectual disability": {
                    "hpo_id": "HP:0001250",
                    "phenotype abnormal category": "Abnormality of the nervous system",
                    "synonyms": [],
                    "parent categories": [],
                    "detailed information": ""
                },
                "Global developmental delay": {
                    "hpo_id": "HP:0001249",
                    "phenotype abnormal category": "Abnormality of the nervous system",
                    "synonyms": [],
                    "parent categories": [],
                    "detailed information": ""
                },
                "Hypotonia": {
                    "hpo_id": "HP:0001252",
                    "phenotype abnormal category": "Abnormality of the nervous system",
                    "synonyms": [],
                    "parent categories": [],
                    "detailed information": ""
                }
            }
        },
        disease_cases={
            "Case 1": {
                "Disease name": "Down syndrome",
                "Disease id": "OMIM:190685",
                "Disease category": "Syndrome",
                "Disease description": "A chromosomal disorder"
            },
            "Case 2": {
                "Disease name": "Rett syndrome",
                "Disease id": "OMIM:312750",
                "Disease category": "Syndrome",
                "Disease description": "A neurodevelopmental disorder"
            }
        }
    )
    print_result("Disease Diagnosis - Full Workflow", result)
    return result

def test_integration_workflow():
    """Test 7: Integration test - Complete workflow"""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST - Complete Workflow")
    print("=" * 80)
    print("This test demonstrates the typical workflow:")
    print("1. Extract phenotypes")
    print("2. Extract disease cases")
    print("3. Retrieve disease information (optional)")
    print("4. Perform diagnosis")
    print("=" * 80)
    print()
    
    query = "A 3-year-old girl with severe intellectual disability, developmental regression, loss of purposeful hand skills, and stereotypic hand movements"
    
    # Step 1: Extract phenotypes
    print("Step 1: Extracting phenotypes...")
    phenotype_result_str = phenotype_extractor(query=query)
    print_result("Integration - Step 1: Extract Phenotypes", phenotype_result_str)
    
    # Parse phenotype result
    try:
        phenotype_result = json.loads(phenotype_result_str)
        extracted_phenotypes = phenotype_result.get("extracted_phenotypes", {})
    except:
        extracted_phenotypes = {}
    
    time.sleep(1)
    
    # Step 2: Extract disease cases
    print("Step 2: Extracting disease cases...")
    disease_cases_result_str = extract_disease_cases(
        query=query,
        top_k=100,
        final_top_k=20
    )
    print_result("Integration - Step 2: Extract Disease Cases", disease_cases_result_str)
    
    # Parse disease cases result
    try:
        disease_cases_result = json.loads(disease_cases_result_str)
        disease_cases = disease_cases_result.get("extracted_disease_cases", {})
    except:
        disease_cases = {}
    
    time.sleep(1)
    
    # Step 3: Retrieve disease information (optional)
    print("Step 3: Retrieving disease information...")
    disease_info_result = disease_information_retrieval(
        query="Rett syndrome, Angelman syndrome",
        use_model=True
    )
    print_result("Integration - Step 3: Retrieve Disease Information", disease_info_result)
    
    time.sleep(1)
    
    # Step 4: Perform diagnosis (if we have valid data)
    if extracted_phenotypes and disease_cases:
        print("Step 4: Performing diagnosis...")
        # Get first phenotype set and first disease case set
        first_phenotype_key = list(extracted_phenotypes.keys())[0] if extracted_phenotypes else None
        first_case_key = list(disease_cases.keys())[0] if disease_cases else None
        
        if first_phenotype_key and first_case_key:
            diagnosis_result = disease_diagnosis(
                original_query=query,
                extracted_phenotypes={first_phenotype_key: extracted_phenotypes[first_phenotype_key]},
                disease_cases={first_case_key: disease_cases[first_case_key]}
            )
            print_result("Integration - Step 4: Perform Diagnosis", diagnosis_result)
    
    print("\n" + "=" * 80)
    print("Integration test completed!")
    print("=" * 80)
    print()

def main():
    """Run all tests"""
    print("Starting Rare Disease Diagnosis MCP Server Tests...")
    print("Testing tool functions directly (not via HTTP)")
    print()
    
    try:
        # Test 2: Phenotype extractor tests
        print("\n" + "#" * 80)
        print("# PHENOTYPE EXTRACTOR TESTS")
        print("#" * 80 + "\n")
        
        test_phenotype_extractor_basic()
        time.sleep(0.5)
        
        test_phenotype_extractor_with_hpo()
        time.sleep(0.5)
        
        test_phenotype_extractor_init_only()
        time.sleep(0.5)
        
        test_phenotype_extractor_full_params()
        time.sleep(0.5)
        
        test_phenotype_extractor_complex()
        time.sleep(0.5)
        
        # Test 3: Disease information retrieval tests
        print("\n" + "#" * 80)
        print("# DISEASE INFORMATION RETRIEVAL TESTS")
        print("#" * 80 + "\n")
        
        test_disease_info_with_model()
        time.sleep(0.5)
        
        test_disease_info_with_id()
        time.sleep(0.5)
        
        test_disease_info_with_candidates()
        time.sleep(0.5)
        
        test_disease_info_multiple()
        time.sleep(0.5)
        
        # Test 4: Extract disease cases tests
        print("\n" + "#" * 80)
        print("# EXTRACT DISEASE CASES TESTS")
        print("#" * 80 + "\n")
        
        test_extract_disease_cases_basic()
        time.sleep(0.5)
        
        test_extract_disease_cases_custom_k()
        time.sleep(0.5)
        
        test_extract_disease_cases_complex()
        time.sleep(0.5)
        
        test_extract_disease_cases_hpo_only()
        time.sleep(0.5)
        
        # Test 5: Disease diagnosis
        print("\n" + "#" * 80)
        print("# DISEASE DIAGNOSIS TESTS")
        print("#" * 80 + "\n")
        
        test_disease_diagnosis_full()
        time.sleep(0.5)
        
        # Test 7: Integration test
        test_integration_workflow()
        
        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

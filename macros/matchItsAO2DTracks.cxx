#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TKey.h>
#include <TList.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <fstream>

struct ITSTrack {
    Int_t event;
    Int_t pdg;
    Float_t pt, eta, phi;
    Long64_t entry; // Keep track of original entry for output
};

struct AO2DTrack {
    Int_t fIndexMcCollisions;
    Int_t fPdgCode;
    Float_t pt, eta, phi;
    Long64_t entry; // Keep track of original entry for output
};

struct Match {
    Long64_t its_entry;
    Long64_t aod_entry;
    Float_t delta_pt, delta_eta, delta_phi;
};

// Hash function for pair<int, int> to use in unordered_map
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Function to calculate angular difference for phi (handle wraparound)
Float_t deltaAngle(Float_t phi1, Float_t phi2) {
    Float_t diff = phi1 - phi2;
    while (diff > M_PI) diff -= 2 * M_PI;
    while (diff <= -M_PI) diff += 2 * M_PI;
    return std::abs(diff);
}

// Function to check if tracks match within tolerances
bool tracksMatch(const ITSTrack& its, const AO2DTrack& aod, 
                Float_t pt_tol = 0.001, Float_t eta_tol = 0.001, Float_t phi_tol = 0.001) {
    Float_t delta_pt = std::abs(its.pt - aod.pt);
    Float_t delta_eta = std::abs(its.eta - aod.eta);
    Float_t delta_phi = deltaAngle(its.phi, aod.phi);
    
    return (delta_pt < pt_tol) && (delta_eta < eta_tol) && (delta_phi < phi_tol);
}

// Function to list all trees in a file
void listTrees(const char* filename) {
    TFile* file = TFile::Open(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    std::cout << "Trees in " << filename << ":" << std::endl;
    TList* keys = file->GetListOfKeys();
    TIter next(keys);
    TKey* key;
    while ((key = (TKey*)next())) {
        if (strcmp(key->GetClassName(), "TTree") == 0) {
            TTree* tree = (TTree*)file->Get(key->GetName());
            std::cout << "  - " << key->GetName() << " (entries: " << tree->GetEntries() << ")" << std::endl;
        }
    }
    file->Close();
}

void matchItsAO2DTracks(const char* its_filename, const char* aod_filename, int tf=1) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Open files
    TFile* its_file = TFile::Open(its_filename, "UPDATE"); // Open in UPDATE mode
    TFile* aod_file = TFile::Open(aod_filename);
    
    if (!its_file || !aod_file) {
        std::cerr << "Error opening files!" << std::endl;
        return;
    }
    
    // Get trees with the specific names you provided
    TTree* its_tree = (TTree*)its_file->Get("RecoTracks");
    std::string aod_tree_name = "DF_" + std::to_string(tf) + "/O2mcparticle_001";
    TTree* aod_tree = (TTree*)aod_file->Get(aod_tree_name.c_str());

    if (!its_tree) {
        std::cout << "Could not find 'RecoTracks' tree in ITS file. Available trees:" << std::endl;
        listTrees(its_filename);
        its_file->Close();
        aod_file->Close();
        return;
    }
    
    if (!aod_tree) {
        std::cout << "Could not find '" << aod_tree_name << "' tree in AO2D file. Available trees:" << std::endl;
        listTrees(aod_filename);
        its_file->Close();
        aod_file->Close();
        return;
    }
    
    std::cout << "ITS tree entries: " << its_tree->GetEntries() << std::endl;
    std::cout << "AO2D tree entries: " << aod_tree->GetEntries() << std::endl;
    
    // Set up branches for ITS tree
    Int_t its_event, its_pdg;
    Float_t its_pt, its_eta, its_phi;
    
    its_tree->SetBranchAddress("event", &its_event);
    its_tree->SetBranchAddress("pdg", &its_pdg);
    its_tree->SetBranchAddress("pt", &its_pt);
    its_tree->SetBranchAddress("eta", &its_eta);
    its_tree->SetBranchAddress("phi", &its_phi);
    
    // Set up branches for AO2D tree
    Int_t aod_event, aod_pdg;
    Float_t  aod_px, aod_py, aod_pz, aod_pt, aod_eta, aod_phi;
    
    aod_tree->SetBranchAddress("fIndexMcCollisions", &aod_event);
    aod_tree->SetBranchAddress("fPdgCode", &aod_pdg);
    aod_tree->SetBranchAddress("fPx", &aod_px);
    aod_tree->SetBranchAddress("fPy", &aod_py);
    aod_tree->SetBranchAddress("fPz", &aod_pz);
    
    // Build index for AO2D tracks: map<(event, pdg), vector<AO2DTrack>>
    std::unordered_map<std::pair<int, int>, std::vector<AO2DTrack>, PairHash> aod_index;
    
    std::cout << "Building AO2D track index..." << std::endl;
    Long64_t aod_entries = aod_tree->GetEntries();
    for (Long64_t i = 0; i < aod_entries; ++i) {
        aod_tree->GetEntry(i);  
        aod_pt = std::sqrt(aod_px * aod_px + aod_py * aod_py);
        aod_eta = 0.5 * std::log((std::sqrt(aod_px * aod_px + aod_py * aod_py + aod_pz * aod_pz ) + aod_pz) /
                                 (std::sqrt(aod_px * aod_px + aod_py * aod_py + aod_pz * aod_pz) - aod_pz));
        aod_phi = std::atan2(aod_py, aod_px);
        if (aod_phi < 0) aod_phi += 2 * M_PI;
        AO2DTrack aod_track = {aod_event, aod_pdg, aod_pt, aod_eta, aod_phi, i};
        aod_index[{aod_event, aod_pdg}].push_back(aod_track);
        
        if (i % 100000 == 0) {
            std::cout << "Processed " << i << " AO2D tracks" << std::endl;
        }
    }
    
    std::cout << "AO2D index built with " << aod_index.size() << " (event, PDG) combinations" << std::endl;
    
    Long64_t its_entries = its_tree->GetEntries(); // Move this line here
    
    // Match ITS tracks to AO2D tracks and store the matches
    std::vector<Long64_t> match_indices(its_entries, -1); // Initialize with -1 (no match)
    int matched_count = 0;
    int total_its = 0;
    
    std::cout << "Matching tracks..." << std::endl;
    
    for (Long64_t i = 0; i < its_entries; ++i) {
        its_tree->GetEntry(i);
        total_its++;
        
        ITSTrack its_track = {its_event, its_pdg, its_pt, its_eta, its_phi, i};
        
        // Look for AO2D tracks with same event and PDG
        auto key = std::make_pair(its_event, its_pdg);
        auto it = aod_index.find(key);
        
        if (it != aod_index.end()) {
            // Check kinematic matching for all candidates
            for (const auto& aod_track : it->second) {
                if (tracksMatch(its_track, aod_track)) {
                    match_indices[i] = aod_track.entry; // Store the matched AO2D entry
                    matched_count++;
                    break; // Take first match
                }
            }
        }
        
        if (i % 50000 == 0) {
            std::cout << "Processed " << i << " ITS tracks, found " << matched_count << " matches" << std::endl;
        }
    }
    
    
    // Now create a new branch in the ITS tree with the matching information
    std::cout << "\nAdding fIndexMcParticles branch to ITS tree..." << std::endl;
    
    its_file->cd();
    TTree* matching_tree = new TTree("MatchingIndex", "AO2D matching indices for ITS tracks");

    // Single branch with the AO2D index for each ITS track
    Long64_t fIndexMcParticles;
    matching_tree->Branch("fIndexMcParticles", &fIndexMcParticles, "fIndexMcParticles/L");

    // Fill the tree with matching indices (same order as ITS tree)
    for (Long64_t i = 0; i < its_entries; ++i) {
        fIndexMcParticles = match_indices[i]; // -1 if no match, AO2D entry if matched
        matching_tree->Fill();
        
        if (i % 100000 == 0) {
            std::cout << "Filled " << i << " entries in matching tree" << std::endl;
        }
    }

    // Write the tree
    std::cout << "Writing MatchingIndex tree to file..." << std::endl;
    matching_tree->Write();

    // Clean up
    delete matching_tree;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== MATCHING RESULTS ===" << std::endl;
    std::cout << "Total ITS tracks: " << total_its << std::endl;
    std::cout << "Total matches found: " << matched_count << std::endl;
    std::cout << "Matching efficiency: " << (100.0 * matched_count / total_its) << "%" << std::endl;
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    
    // Show some examples of the matching
    std::cout << "\nFirst 10 entries with their fIndexMcParticles values:" << std::endl;
    std::cout << "ITS_Entry\tfIndexMcParticles" << std::endl;
    for (size_t i = 0; i < std::min(10UL, (size_t)its_entries); ++i) {
        std::cout << i << "\t\t" << match_indices[i] << std::endl;
    }
    
    std::cout << "\nBranch 'fIndexMcParticles' successfully added to RecoTracks tree!" << std::endl;
    std::cout << "Values: -1 = no match found, >= 0 = AO2D entry number of matched particle" << std::endl;
    
    // // Optional: Save results to a file
    // std::cout << "\nSaving results to matches.txt..." << std::endl;
    // std::ofstream outfile("matches.txt");
    // outfile << "its_entry aod_entry delta_pt delta_eta delta_phi" << std::endl;
    // for (const auto& match : matches) {
    //     outfile << match.its_entry << " " << match.aod_entry << " "
    //             << match.delta_pt << " " << match.delta_eta << " "
    //             << match.delta_phi << std::endl;
    // }
    // outfile.close();
    
    // Clean up
    its_file->Close();
    aod_file->Close();
}
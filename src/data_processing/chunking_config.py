"""
Configuration management cho chunking system.
Cung cấp các preset configurations cho QuantConnect documentation.
"""

from pathlib import Path
from typing import Dict, Optional
import json
import yaml

from src.data_processing.chunk_models import ChunkingConfig, ChunkingStrategy


class QuantConnectChunkingConfig:
    """
    Specialized chunking configurations cho QuantConnect documentation.
    Tối ưu cho mixed content (text + code + tables).
    """
    
    @staticmethod
    def default() -> ChunkingConfig:
        """Default configuration cho QuantConnect docs"""
        return ChunkingConfig(
            # Size settings - balanced cho cả text và code
            max_chunk_size=1200,
            min_chunk_size=200,
            chunk_overlap=150,
            
            # Token limits (for embedding models)
            max_chunk_tokens=384,  # Good for most embedding models
            chunk_overlap_tokens=50,
            
            # Strategy
            default_strategy=ChunkingStrategy.HYBRID,
            
            # Code handling - QuantConnect có nhiều code examples
            preserve_small_code_blocks=True,
            max_code_block_size=1500,
            code_chunk_by_function=True,
            
            # Section handling - respect document structure
            respect_section_boundaries=True,
            include_section_header_in_chunks=True,
            create_section_summary_chunk=True,
            
            # Table handling
            preserve_small_tables=True,
            max_table_rows_intact=20,
            
            # Text processing
            sentence_splitter_regex=r'(?<=[.!?])\s+(?=[A-Z])',
            paragraph_splitter='\n\n',
            
            # Quality settings
            ensure_complete_sentences=True,
            ensure_complete_code_blocks=True,
            
            # Metadata
            include_chunk_position_meta=True,
            include_hierarchy_meta=True
        )
    
    @staticmethod
    def for_python_content() -> ChunkingConfig:
        """Configuration optimized cho Python code content"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 1500  # Larger for code
        config.max_code_block_size = 2000
        config.code_chunk_by_function = True
        config.default_strategy = ChunkingStrategy.CODE_AWARE
        return config
    
    @staticmethod
    def for_csharp_content() -> ChunkingConfig:
        """Configuration optimized cho C# code content"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 1500  # Larger for code
        config.max_code_block_size = 2000
        config.code_chunk_by_function = True
        config.default_strategy = ChunkingStrategy.CODE_AWARE
        # C# often has longer class definitions
        config.max_chunk_size = 1800
        return config
    
    @staticmethod
    def for_api_reference() -> ChunkingConfig:
        """Configuration cho API reference sections"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 800  # Smaller chunks for precise retrieval
        config.chunk_overlap = 100
        config.create_section_summary_chunk = True
        config.respect_section_boundaries = True
        return config
    
    @staticmethod
    def for_tutorial_content() -> ChunkingConfig:
        """Configuration cho tutorial/guide content"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 1000
        config.chunk_overlap = 200  # More overlap for context continuity
        config.include_section_header_in_chunks = True
        config.default_strategy = ChunkingStrategy.PARAGRAPH_BASED
        return config
    
    @staticmethod
    def for_lean_engine() -> ChunkingConfig:
        """Specific configuration cho Lean Engine documentation"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 1200
        config.preserve_small_code_blocks = True
        config.max_code_block_size = 1800  # Lean Engine có complex code examples
        return config
    
    @staticmethod
    def for_writing_algorithms() -> ChunkingConfig:
        """Specific configuration cho Writing Algorithms documentation"""
        config = QuantConnectChunkingConfig.default()
        config.max_chunk_size = 1000
        config.chunk_overlap = 200
        # Writing Algorithms có nhiều step-by-step guides
        config.default_strategy = ChunkingStrategy.PARAGRAPH_BASED
        config.include_section_header_in_chunks = True
        return config


class ChunkingConfigManager:
    """
    Manages chunking configurations.
    Supports loading/saving configs and selecting appropriate config based on content.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/chunking")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache loaded configs
        self._configs: Dict[str, ChunkingConfig] = {}
        
        # Initialize default configs
        self._init_default_configs()
    
    def _init_default_configs(self):
        """Initialize default configurations"""
        self._configs.update({
            'default': QuantConnectChunkingConfig.default(),
            'python': QuantConnectChunkingConfig.for_python_content(),
            'csharp': QuantConnectChunkingConfig.for_csharp_content(),
            'api_reference': QuantConnectChunkingConfig.for_api_reference(),
            'tutorial': QuantConnectChunkingConfig.for_tutorial_content(),
            'lean_engine': QuantConnectChunkingConfig.for_lean_engine(),
            'writing_algorithms': QuantConnectChunkingConfig.for_writing_algorithms(),
        })
    
    def get_config(self, name: str = 'default') -> ChunkingConfig:
        """Get configuration by name"""
        if name in self._configs:
            return self._configs[name]
        
        # Try to load from file
        config_file = self.config_dir / f"{name}.json"
        if config_file.exists():
            return self.load_config(name)
        
        # Fallback to default
        return self._configs['default']
    
    def get_config_for_file(self, file_name: str) -> ChunkingConfig:
        """Get appropriate config based on file name"""
        file_lower = file_name.lower()
        
        if 'lean-engine' in file_lower:
            return self.get_config('lean_engine')
        elif 'writing-algorithms' in file_lower:
            return self.get_config('writing_algorithms')
        elif 'lean-cli' in file_lower:
            return self.get_config('api_reference')
        elif 'research' in file_lower:
            return self.get_config('tutorial')
        
        return self.get_config('default')
    
    def get_config_for_section(
        self, 
        section_title: str, 
        has_code: bool = False,
        code_language: Optional[str] = None
    ) -> ChunkingConfig:
        """Get appropriate config based on section characteristics"""
        title_lower = section_title.lower()
        
        # Check for specific section types
        if any(keyword in title_lower for keyword in ['api', 'reference', 'class', 'method']):
            return self.get_config('api_reference')
        
        if any(keyword in title_lower for keyword in ['tutorial', 'guide', 'getting started', 'how to']):
            return self.get_config('tutorial')
        
        # Check for code-heavy sections
        if has_code:
            if code_language == 'python':
                return self.get_config('python')
            elif code_language == 'csharp':
                return self.get_config('csharp')
        
        return self.get_config('default')
    
    def save_config(self, name: str, config: ChunkingConfig):
        """Save configuration to file"""
        config_file = self.config_dir / f"{name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Update cache
        self._configs[name] = config
    
    def load_config(self, name: str) -> ChunkingConfig:
        """Load configuration from file"""
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Convert strategy string back to enum
        if 'default_strategy' in config_dict:
            config_dict['default_strategy'] = ChunkingStrategy(config_dict['default_strategy'])
        
        config = ChunkingConfig(**config_dict)
        
        # Update cache
        self._configs[name] = config
        
        return config
    
    def list_configs(self) -> Dict[str, str]:
        """List all available configurations"""
        configs = {}
        
        # Built-in configs
        for name in self._configs:
            configs[name] = "Built-in configuration"
        
        # File-based configs
        for config_file in self.config_dir.glob("*.json"):
            name = config_file.stem
            if name not in configs:
                configs[name] = f"Custom configuration from {config_file.name}"
        
        return configs
    
    def export_all_configs(self, export_dir: Optional[Path] = None):
        """Export all configurations to files"""
        export_dir = export_dir or self.config_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for name, config in self._configs.items():
            self.save_config(name, config)
        
        print(f"Exported {len(self._configs)} configurations to {export_dir}")


# Singleton instance
_config_manager = ChunkingConfigManager()


def get_chunking_config(name: str = 'default') -> ChunkingConfig:
    """Get chunking configuration by name"""
    return _config_manager.get_config(name)


def get_chunking_config_for_file(file_name: str) -> ChunkingConfig:
    """Get appropriate chunking config for a file"""
    return _config_manager.get_config_for_file(file_name)


def get_chunking_config_for_section(
    section_title: str,
    has_code: bool = False,
    code_language: Optional[str] = None
) -> ChunkingConfig:
    """Get appropriate chunking config for a section"""
    return _config_manager.get_config_for_section(section_title, has_code, code_language)


# Example usage
if __name__ == "__main__":
    # Test config manager
    manager = ChunkingConfigManager()
    
    # List available configs
    print("Available configurations:")
    for name, desc in manager.list_configs().items():
        print(f"  - {name}: {desc}")
    
    # Get config for specific file
    config = manager.get_config_for_file("Quantconnect-Writing-Algorithms.html")
    print(f"\nConfig for Writing Algorithms:")
    print(f"  Max chunk size: {config.max_chunk_size}")
    print(f"  Strategy: {config.default_strategy.value}")
    
    # Get config for code-heavy section
    config = manager.get_config_for_section("Python Code Examples", has_code=True, code_language="python")
    print(f"\nConfig for Python code section:")
    print(f"  Max chunk size: {config.max_chunk_size}")
    print(f"  Max code block size: {config.max_code_block_size}")
    
    # Export all configs
    manager.export_all_configs()
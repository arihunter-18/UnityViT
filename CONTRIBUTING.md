# Contributing to SHViT

Thank you for your interest in contributing to SHViT! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:
1. Check if the issue already exists in the [Issues](https://github.com/ysj9909/SHViT/issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, PyTorch version, GPU)

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SHViT.git
   cd SHViT
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   - Ensure existing tests pass
   - Add new tests for new features
   - Verify no performance regression

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference related issues
   - Wait for code review

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and concise

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ysj9909/SHViT.git
cd SHViT

# Create development environment
conda create -n shvit-dev python=3.9
conda activate shvit-dev

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8
```

## Testing

Before submitting a PR, ensure:
- All existing tests pass
- New features include appropriate tests
- Code follows style guidelines

```bash
# Run tests
python -m pytest tests/

# Format code
black .

# Check style
flake8 .
```

## Areas for Contribution

We especially welcome contributions in:
- **Documentation**: Improve tutorials, add examples
- **Performance**: Optimization, faster inference
- **Features**: New model variants, training strategies
- **Robustness**: Additional attack methods, defense mechanisms
- **Deployment**: Mobile optimization, quantization
- **Bug fixes**: Fix reported issues

## Questions?

Feel free to:
- Open a discussion in the [Discussions](https://github.com/ysj9909/SHViT/discussions) tab
- Ask questions in issues
- Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment

Thank you for contributing to SHViT! 🚀


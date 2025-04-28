// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Address.sol";

contract FlashArbitrage is ReentrancyGuard, Ownable {
    // Mapping of token addresses to their decimals
    mapping(address => uint8) public tokenDecimals;
    
    // Event emitted when an arbitrage is executed
    event ArbitrageExecuted(
        address indexed executor,
        address[] path,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit
    );
    
    // Event emitted when a token is added
    event TokenAdded(address indexed token, uint8 decimals);
    
    // Event emitted when a token is removed
    event TokenRemoved(address indexed token);
    
    // Mapping of allowed tokens
    mapping(address => bool) public allowedTokens;
    
    // Mapping of token pairs to their reserves
    mapping(address => mapping(address => uint256)) public reserves;
    
    // Constructor
    constructor() ReentrancyGuard() Ownable() {
        // Initialize with common tokens
        _addToken(0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2, 18); // WETH
        _addToken(0xdAC17F958D2ee523a2206206994597C13D831ec7, 6);  // USDT
        _addToken(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48, 6);  // USDC
    }
    
    // Add a new token to the allowed list
    function addToken(address token, uint8 decimals) external onlyOwner {
        require(token != address(0), "Invalid token address");
        require(!allowedTokens[token], "Token already added");
        
        _addToken(token, decimals);
    }

    function getTokenPairs() external view returns (address[] memory pairs) {
        uint256 count = 0;
        address[] memory tempPairs = new address[](256); // Temporary array for token pairs

        for (uint i = 0; i < allowedTokensList.length; i++) {
        address tokenA = allowedTokensList[i];
        for (uint j = 0; j < allowedTokensList.length; j++) {
            address tokenB = allowedTokensList[j];
            if (tokenA != tokenB && reserves[tokenA][tokenB] > 0) {
                tempPairs[count] = tokenA;
                tempPairs[count + 1] = tokenB;
                count += 2;
            }
        }
    }

    // Resize the array to fit the actual number of pairs
    pairs = new address[](count);
    for (uint256 i = 0; i < count; i++) {
        pairs[i] = tempPairs[i];
    }

    return pairs;
}

    // Remove a token from the allowed list
    function removeToken(address token) external onlyOwner {
        require(allowedTokens[token], "Token not found");
        allowedTokens[token] = false;
        emit TokenRemoved(token);
    }
    
    // Execute an arbitrage trade
    function executeArbitrage(
        address[] calldata path,
        uint256 amountIn
    ) external nonReentrant returns (uint256 amountOut) {
        require(path.length >= 2, "Path must have at least 2 tokens");
        
        // Get the current reserves
        uint256[] memory amounts = getAmountsIn(amountIn, path);
        amountOut = amounts[amounts.length - 1];
        
        // Calculate profit
        uint256 profit = amountOut - amountIn;
        require(profit > 0, "No profit");
        
        // Execute the trades
        _executeTrades(path, amounts);
        
        // Emit event
        emit ArbitrageExecuted(
            msg.sender,
            path,
            amountIn,
            amountOut,
            profit
        );
        
        return amountOut;
    }
    
    // Get the amounts out for a given path
    function getAmountsIn(uint256 amountIn, address[] calldata path)
        public
        view
        returns (uint256[] memory amounts)
    {
        require(path.length >= 2, "Path must have at least 2 tokens");
        
        amounts = new uint256[](path.length);
        amounts[0] = amountIn;
        
        for (uint256 i = 0; i < path.length - 1; i++) {
            amounts[i + 1] = _getAmountOut(
                amounts[i],
                path[i],
                path[i + 1]
            );
        }
        
        return amounts;
    }
    
    // Get amount out for a single swap
    function _getAmountOut(
        uint256 amountIn,
        address tokenIn,
        address tokenOut
    ) internal view returns (uint256 amountOut) {
        require(allowedTokens[tokenIn], "TokenIn not allowed");
        require(allowedTokens[tokenOut], "TokenOut not allowed");
        
        uint256 reserveIn = reserves[tokenIn][tokenOut];
        uint256 reserveOut = reserves[tokenOut][tokenIn];
        
        uint256 amountInWithFee = amountIn * 997;
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = reserveIn * 1000 + amountInWithFee;
        
        amountOut = numerator / denominator;
        
        return amountOut;
    }
    
    // Execute the trades
    function _executeTrades(
        address[] memory path,
        uint256[] memory amounts
    ) internal {
        for (uint256 i = 0; i < path.length - 1; i++) {
            address tokenIn = path[i];
            address tokenOut = path[i + 1];
            uint256 amount = amounts[i];
            
            // Update reserves
            reserves[tokenIn][tokenOut] -= amount;
            reserves[tokenOut][tokenIn] += amounts[i + 1];
        }
    }
    
    // Internal function to add a token
    function _addToken(address token, uint8 decimals) internal {
        require(token != address(0), "Invalid token address");
        allowedTokens[token] = true;
        tokenDecimals[token] = decimals;
        emit TokenAdded(token, decimals);
    }
}
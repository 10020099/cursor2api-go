package services

import (
	"cursor2api-go/config"
	"cursor2api-go/models"
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildCursorRequestEnablesToolProtocolForBaseModel(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			SystemPromptInject: "Injected system prompt",
			MaxInputLength:     10000,
		},
	}

	request := &models.ChatCompletionRequest{
		Model: "claude-sonnet-4.6",
		Messages: []models.Message{
			{Role: "user", Content: "What's the weather?"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.FunctionDefinition{
					Name:        "get_weather",
					Description: "Fetch current weather",
					Parameters: map[string]interface{}{
						"type": "object",
					},
				},
			},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}

	if result.Payload.Model != "anthropic/claude-sonnet-4.6" {
		t.Fatalf("Payload.Model = %v, want anthropic/claude-sonnet-4.6", result.Payload.Model)
	}
	if result.ParseConfig.TriggerSignal == "" {
		t.Fatalf("TriggerSignal should not be empty")
	}
	if result.ParseConfig.ThinkingEnabled {
		t.Fatalf("ThinkingEnabled = true, want false")
	}

	systemText := result.Payload.Messages[0].Parts[0].Text
	if !strings.Contains(systemText, "<function_list>") {
		t.Fatalf("system prompt does not include function list: %s", systemText)
	}
	if strings.Contains(systemText, thinkingHint) {
		t.Fatalf("system prompt should not include thinking hint for base model")
	}
}

func TestBuildCursorRequestThinkingModelFormatsToolHistory(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			MaxInputLength: 10000,
		},
	}

	request := &models.ChatCompletionRequest{
		Model: "claude-sonnet-4.6-thinking",
		Messages: []models.Message{
			{Role: "user", Content: "Plan first, then use tools."},
			{
				Role: "assistant",
				ToolCalls: []models.ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: models.FunctionCall{
							Name:      "lookup",
							Arguments: `{"q":"revivalquant"}`,
						},
					},
				},
			},
			{Role: "tool", ToolCallID: "call_1", Name: "lookup", Content: "Found result"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.FunctionDefinition{
					Name: "lookup",
				},
			},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}

	if result.ParseConfig.TriggerSignal == "" {
		t.Fatalf("TriggerSignal should not be empty")
	}
	if !result.ParseConfig.ThinkingEnabled {
		t.Fatalf("ThinkingEnabled = false, want true")
	}
	if result.Payload.Model != "anthropic/claude-sonnet-4.6" {
		t.Fatalf("Payload.Model = %v, want anthropic/claude-sonnet-4.6", result.Payload.Model)
	}

	userText := result.Payload.Messages[1].Parts[0].Text
	if !strings.Contains(userText, thinkingHint) {
		t.Fatalf("user message should contain thinking hint, got: %s", userText)
	}

	assistantText := result.Payload.Messages[2].Parts[0].Text
	if !strings.Contains(assistantText, result.ParseConfig.TriggerSignal) {
		t.Fatalf("assistant tool history should include trigger signal, got: %s", assistantText)
	}
	if !strings.Contains(assistantText, `<invoke name="lookup">{"q":"revivalquant"}</invoke>`) {
		t.Fatalf("assistant tool history missing invoke block, got: %s", assistantText)
	}

	toolText := result.Payload.Messages[3].Parts[0].Text
	if !strings.Contains(toolText, `<tool_result id="call_1" name="lookup">Found result</tool_result>`) {
		t.Fatalf("tool result history missing tool_result block, got: %s", toolText)
	}
}

func TestBuildCursorRequestPreservesToolHistoryWithoutCurrentTools(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			MaxInputLength: 10000,
		},
	}

	request := &models.ChatCompletionRequest{
		Model:      "claude-sonnet-4.6",
		ToolChoice: []byte(`"none"`),
		Messages: []models.Message{
			{
				Role: "assistant",
				ToolCalls: []models.ToolCall{
					{
						ID:   "call_weather",
						Type: "function",
						Function: models.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"city":"Beijing"}`,
						},
					},
				},
			},
			{Role: "tool", ToolCallID: "call_weather", Name: "get_weather", Content: "Sunny"},
			{Role: "user", Content: "Summarize the result."},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}

	if result.ParseConfig.TriggerSignal == "" {
		t.Fatalf("TriggerSignal should be kept for tool history replay")
	}

	systemText := result.Payload.Messages[0].Parts[0].Text
	if !strings.Contains(systemText, "completed history") {
		t.Fatalf("system prompt should explain historical tool transcript, got: %s", systemText)
	}
	if !strings.Contains(systemText, result.ParseConfig.TriggerSignal) {
		t.Fatalf("system prompt should include trigger signal, got: %s", systemText)
	}

	assistantText := result.Payload.Messages[1].Parts[0].Text
	if !strings.Contains(assistantText, result.ParseConfig.TriggerSignal) {
		t.Fatalf("assistant history should preserve trigger signal, got: %s", assistantText)
	}
}

func TestBuildCursorRequestAllowsToolChoiceNoneWithoutTools(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			MaxInputLength: 10000,
		},
	}

	request := &models.ChatCompletionRequest{
		Model:      "claude-sonnet-4.6",
		ToolChoice: []byte(`"none"`),
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}
	if result.ParseConfig.TriggerSignal != "" {
		t.Fatalf("TriggerSignal = %q, want empty for plain chat", result.ParseConfig.TriggerSignal)
	}
	if len(result.Payload.Messages) != 1 {
		t.Fatalf("payload message count = %d, want 1", len(result.Payload.Messages))
	}
}

func TestBuildCursorRequestCountsSerializedToolCallsInMaxInputLength(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			MaxInputLength: 20,
		},
	}

	request := &models.ChatCompletionRequest{
		Model: "claude-sonnet-4.6",
		Messages: []models.Message{
			{
				Role: "assistant",
				ToolCalls: []models.ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: models.FunctionCall{
							Name:      "lookup",
							Arguments: `{"payload":"1234567890123456789012345678901234567890"}`,
						},
					},
				},
			},
			{Role: "user", Content: "Short"},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}

	for _, msg := range result.Payload.Messages {
		if strings.Contains(msg.Parts[0].Text, `"payload":"1234567890123456789012345678901234567890"`) {
			t.Fatalf("serialized tool call arguments should be removed by truncation, payload still contains long tool json: %#v", result.Payload.Messages)
		}
	}
	totalLength := 0
	for _, msg := range result.Payload.Messages {
		totalLength += len(msg.Parts[0].Text)
	}
	if totalLength == 0 {
		t.Fatalf("truncation should preserve at least one message")
	}
}

func TestParseToolChoiceObjectForm(t *testing.T) {
	tests := []struct {
		name     string
		raw      json.RawMessage
		wantMode string
		wantErr  bool
	}{
		{
			name:     "object auto",
			raw:      []byte(`{"type":"auto"}`),
			wantMode: "auto",
		},
		{
			name:     "object none",
			raw:      []byte(`{"type":"none"}`),
			wantMode: "none",
		},
		{
			name:     "object required",
			raw:      []byte(`{"type":"required"}`),
			wantMode: "required",
		},
		{
			name:     "string auto",
			raw:      []byte(`"auto"`),
			wantMode: "auto",
		},
		{
			name:     "object function",
			raw:      []byte(`{"type":"function","function":{"name":"get_weather"}}`),
			wantMode: "function",
		},
		{
			name:    "object unsupported type",
			raw:     []byte(`{"type":"unknown"}`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseToolChoice(tt.raw)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseToolChoice() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && got.Mode != tt.wantMode {
				t.Fatalf("parseToolChoice() mode = %q, want %q", got.Mode, tt.wantMode)
			}
		})
	}
}

func TestBuildCursorRequestToolChoiceObjectAutoWithTools(t *testing.T) {
	service := &CursorService{
		config: &config.Config{
			MaxInputLength: 10000,
		},
	}

	request := &models.ChatCompletionRequest{
		Model:      "claude-sonnet-4.6",
		ToolChoice: []byte(`{"type":"auto"}`),
		Messages: []models.Message{
			{Role: "user", Content: "What's the weather?"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.FunctionDefinition{
					Name:        "get_weather",
					Description: "Fetch current weather",
					Parameters:  map[string]interface{}{"type": "object"},
				},
			},
		},
	}

	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() error = %v", err)
	}
	if result.ParseConfig.TriggerSignal == "" {
		t.Fatalf("TriggerSignal should not be empty when tools are provided with auto")
	}
}

func TestFilterValidToolsSkipsNonFunctionAndEmptyName(t *testing.T) {
	tools := []models.Tool{
		{Type: "function", Function: models.FunctionDefinition{Name: "get_weather"}},
		{Type: "code_interpreter"},
		{Type: "function", Function: models.FunctionDefinition{Name: ""}},
		{Type: "file_search"},
		{Type: "function", Function: models.FunctionDefinition{Name: "search"}},
	}

	filtered := filterValidTools(tools)
	if len(filtered) != 2 {
		t.Fatalf("filterValidTools() returned %d tools, want 2", len(filtered))
	}
	if filtered[0].Function.Name != "get_weather" {
		t.Fatalf("filtered[0] = %q, want get_weather", filtered[0].Function.Name)
	}
	if filtered[1].Function.Name != "search" {
		t.Fatalf("filtered[1] = %q, want search", filtered[1].Function.Name)
	}
}

func TestFilterValidToolsDeduplicates(t *testing.T) {
	tools := []models.Tool{
		{Type: "function", Function: models.FunctionDefinition{Name: "get_weather", Description: "first"}},
		{Type: "function", Function: models.FunctionDefinition{Name: "get_weather", Description: "dup"}},
	}
	filtered := filterValidTools(tools)
	if len(filtered) != 1 {
		t.Fatalf("filterValidTools() returned %d tools, want 1", len(filtered))
	}
	if filtered[0].Function.Description != "first" {
		t.Fatalf("first-wins dedup failed: got %q", filtered[0].Function.Description)
	}
}

func TestFilterValidToolsEmptyTypeDefaultsToFunction(t *testing.T) {
	tools := []models.Tool{
		{Function: models.FunctionDefinition{Name: "lookup"}},
	}
	filtered := filterValidTools(tools)
	if len(filtered) != 1 || filtered[0].Function.Name != "lookup" {
		t.Fatalf("empty type tool should be kept, got %v", filtered)
	}
}

func TestBuildCursorRequestSkipsInvalidToolsWithoutError(t *testing.T) {
	service := &CursorService{
		config: &config.Config{MaxInputLength: 10000},
	}
	request := &models.ChatCompletionRequest{
		Model: "claude-sonnet-4.6",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		Tools: []models.Tool{
			{Type: "function", Function: models.FunctionDefinition{Name: "get_weather"}},
			{Type: "code_interpreter"},
			{Type: "function", Function: models.FunctionDefinition{Name: ""}},
		},
	}
	result, err := service.buildCursorRequest(request)
	if err != nil {
		t.Fatalf("buildCursorRequest() should not error for mixed tools, got: %v", err)
	}
	systemText := result.Payload.Messages[0].Parts[0].Text
	if !strings.Contains(systemText, "get_weather") {
		t.Fatalf("system prompt should contain get_weather, got: %s", systemText)
	}
}

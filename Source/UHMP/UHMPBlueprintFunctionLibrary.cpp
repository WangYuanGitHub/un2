// Fill out your copyright notice in the Description page of Project Settings.


#include "UHMPBlueprintFunctionLibrary.h"

void UUHMPBlueprintFunctionLibrary::PrintStringSpecial(FString string)
{
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Blue, string);
	}
}


void UUHMPBlueprintFunctionLibrary::AssertFalse(FString string)
{
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Blue, string);
	}
	check(false);
}

void UUHMPBlueprintFunctionLibrary::RaiseError(UObject* WorldContextObject, const FString& ErrorMessage, bool bPrintToOutputLog)
{
	FString MessageToLog = FString::Printf(TEXT("\"%s\""), *ErrorMessage);


#if WITH_EDITOR
	/*FKismetDebugUtilities::OnScriptException(WorldContextObject,);*/
	TSharedRef<FTokenizedMessage> TokenizedMessage = FTokenizedMessage::Create(EMessageSeverity::Error, FText::FromString(MessageToLog));
	TokenizedMessage.Get().AddToken(FUObjectToken::Create(WorldContextObject));
	FMessageLog BlueprintLog = FMessageLog("BlueprintLog").SuppressLoggingToOutputLog(!bPrintToOutputLog);
	BlueprintLog.AddMessage(TokenizedMessage);
	BlueprintLog.Notify();

#else
	if (bPrintToOutputLog)
	{
		UE_LOG(LogTemp, Error, TEXT("%s"), *MessageToLog);
	}
#endif
}


void UUHMPBlueprintFunctionLibrary::RaiseFatalError(UObject* WorldContextObject, const FString& ErrorMessage)
{
	FString MessageToLog = FString::Printf(TEXT("\"%s\""), *ErrorMessage);
	//char* result = TCHAR_TO_ANSI(*ErrorMessage);
	//assertm(false, result);


#if WITH_EDITOR
	/*FKismetDebugUtilities::OnScriptException(WorldContextObject,);*/
	TSharedRef<FTokenizedMessage> TokenizedMessage = FTokenizedMessage::Create(EMessageSeverity::Error, FText::FromString(MessageToLog));
	TokenizedMessage.Get().AddToken(FUObjectToken::Create(WorldContextObject));
	FMessageLog BlueprintLog = FMessageLog("BlueprintLog").SuppressLoggingToOutputLog(false);
	BlueprintLog.AddMessage(TokenizedMessage);
	BlueprintLog.Notify();
#else
	UE_LOG(LogTemp, Error, TEXT("%s"), *MessageToLog);
	//ensureAlwaysMsgf(false, TEXT("%s"), *Messa geToLog);
	checkf(false, TEXT("%s"), *MessageToLog);

#endif
}


FVector UUHMPBlueprintFunctionLibrary::VectorClearZ(FVector Vector)
{
	FVector new_vec = FVector(Vector);
	new_vec.Z = 0;
	return new_vec;
}


bool UUHMPBlueprintFunctionLibrary::IsAllZeroVector(FVector Vector)
{
	if (Vector.X == 0 && Vector.Z == 0 && Vector.Y == 0) return true;
	else return false;
}


bool UUHMPBlueprintFunctionLibrary::IsZero(int x)
{
	if (x == 0) return true;
	else return false;
}

bool UUHMPBlueprintFunctionLibrary::IsZeroFloat(float x)
{
	if (x == 0) return true;
	else return false;
}


float UUHMPBlueprintFunctionLibrary::GetFrameRatePerGameSecond()
{
	return GEngine->FixedFrameRate;
}

float UUHMPBlueprintFunctionLibrary::GetSimDeltaTime(const UObject* WorldContext)
{
	return UGameplayStatics::GetGlobalTimeDilation(WorldContext) / GEngine->FixedFrameRate;
}

void UUHMPBlueprintFunctionLibrary::SetAITeamForPerceptionFilter(AAIController* Controller, const FGenericTeamId& NewTeamID)
{
	Controller->SetGenericTeamId(NewTeamID);
}


FGenericTeamId UUHMPBlueprintFunctionLibrary::GetAITeamForPerceptionFilter(AAIController* Controller)
{
	return Controller->GetGenericTeamId();
}

TArray<int> UUHMPBlueprintFunctionLibrary::GetAffilationArray(const TArray<AAgentBaseCpp*>& agents)
{
	TArray<int> AffilationArray;
	for (auto agent : agents)
	{
		if (agent) 
		{
			int teamID = agent->GenericTeamNo;
			ensureMsgf(teamID>=0, TEXT("Team ID must > 0"));
			AffilationArray.Add(teamID);
		}
		else 
		{
			AffilationArray.Add(-1);
		}
	}
	return AffilationArray;
}

void UUHMPBlueprintFunctionLibrary::MannualGc()
{
	// void UEngine::ForceGarbageCollection(bool bForcePurge/*=false*/)
	GEngine->ForceGarbageCollection(true);
}

bool UUHMPBlueprintFunctionLibrary::IsEditor()
{
	return GEngine->IsEditor();
}
 
	

FVector UUHMPBlueprintFunctionLibrary::FlyingTracking(FVector self_pos, FVector dst_pos, bool maintain_z, float dis_aim)
{
	// get delta vector and height error
	FVector delta_vec = self_pos - dst_pos;
	float z_delta = FMath::Abs(delta_vec.Z);

	// get horizontal distance to satisfy dis_aim
	float horizontal_dis_aim = 0;
	if (dis_aim > z_delta) 
	{
		horizontal_dis_aim = FMath::Sqrt(dis_aim * dis_aim - z_delta * z_delta);
	}
	else
	{
		// PrintStringSpecial("z_delta > dis_aim !");
	}
	// remove z, and normalize
	delta_vec.Z = 0;
	delta_vec.Normalize();

	// not sure Fvector is passed as reference or value, so copy it first
	FVector dst_posx;
	dst_posx = dst_pos;
	if (maintain_z)
	{
		dst_posx.Z = self_pos.Z;
	}
	dst_posx = dst_posx + delta_vec * horizontal_dis_aim;

	return dst_posx;
}


//void bubble_sort(T arr[], int len) {
//	int i, j;
//	for (i = 0; i < len - 1; i++)
//		for (j = 0; j < len - 1 - i; j++)
//			if (arr[j] > arr[j + 1])
//				swap(arr[j], arr[j + 1]);
//}

void UUHMPBlueprintFunctionLibrary::SortActorListBy(TArray<AActor*> InActors, TArray<float> ScoreList, TArray<AActor*>& OutActors)
{
	OutActors.Reset();
	int i, j;
	int len = InActors.Num();
	for (i = 0; i < len - 1; i++)
	{
		for (j = 0; j < len - 1 - i; j++)
		{
			if (ScoreList[j] > ScoreList[j + 1]) {
				ScoreList.Swap(j, j + 1);
				InActors.Swap(j, j + 1);
			}
		}
	}
	OutActors = InActors;
}


//void UUHMPBlueprintFunctionLibrary::MyGetAllActorsOfClass(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors)
//{
//	OutActors.Reset();
//
//	// We do nothing if no is class provided, rather than giving ALL actors!
//	if (ActorClass)
//	{
//		if (UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull))
//		{
//			for (TActorIterator<AActor> It(World, ActorClass); It; ++It)
//			{
//				AActor* Actor = *It;
//				OutActors.Add(Actor);
//			}
//		}
//	}
//}
//

void UUHMPBlueprintFunctionLibrary::TarrayChangeClass(TArray<AActor*> InActors, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors)
{
	OutActors = InActors;
}


void UUHMPBlueprintFunctionLibrary::GetAllActorsOfClassWithOrder(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors)
{
	QUICK_SCOPE_CYCLE_COUNTER(UGameplayStatics_GetAllActorsOfClass);
	OutActors.Reset();

	// We do nothing if no is class provided, rather than giving ALL actors!
	if (ActorClass)
	{
		if (UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull))
		{
			for (TActorIterator<AActor> It(World, ActorClass); It; ++It)
			{
				AActor* Actor = *It;
				OutActors.Add(Actor);
			}
			// Sort the actors according to their z-axis location
			OutActors.Sort([](const AActor& Actor1, const AActor& Actor2)
				{
					if (Actor1.GetActorLocation().X != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().X;
					if (Actor1.GetActorLocation().Y != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().Y;
					// if (Actor1.GetActorLocation().Z != Actor2.GetActorLocation().Z) return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
					return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
				});

		}
	}
}


void UUHMPBlueprintFunctionLibrary::GetAllActorsWithTagWithOrder(const UObject* WorldContextObject, FName Tag, TArray<AActor*>& OutActors)
{
	QUICK_SCOPE_CYCLE_COUNTER(UGameplayStatics_GetAllActorsWithTag);
	OutActors.Reset();

	// We do nothing if no tag is provided, rather than giving ALL actors!
	if (!Tag.IsNone())
	{
		if (UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull))
		{
			for (FActorIterator It(World); It; ++It)
			{
				AActor* Actor = *It;
				if (Actor->ActorHasTag(Tag))
				{
					OutActors.Add(Actor);
				}
			}
			// Sort the actors according to their z-axis location
			OutActors.Sort([](const AActor& Actor1, const AActor& Actor2)
				{
					if (Actor1.GetActorLocation().X != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().X;
					if (Actor1.GetActorLocation().Y != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().Y;
					// if (Actor1.GetActorLocation().Z != Actor2.GetActorLocation().Z) return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
					return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
				});

		}
	}
}


void UUHMPBlueprintFunctionLibrary::GetAllActorsOfClassWithTagWithOrder(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, FName Tag, TArray<AActor*>& OutActors)
{
	QUICK_SCOPE_CYCLE_COUNTER(UGameplayStatics_GetAllActorsOfClass);
	OutActors.Reset();

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);

	// We do nothing if no is class provided, rather than giving ALL actors!
	if (ActorClass && World)
	{
		for (TActorIterator<AActor> It(World, ActorClass); It; ++It)
		{
			AActor* Actor = *It;
			if (Actor && !Actor->IsPendingKill() && Actor->ActorHasTag(Tag))
			{
				OutActors.Add(Actor);
			}
		}
		// Sort the actors according to their z-axis location
		OutActors.Sort([](const AActor& Actor1, const AActor& Actor2)
			{
				if (Actor1.GetActorLocation().X != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().X;
				if (Actor1.GetActorLocation().Y != Actor2.GetActorLocation().X) return Actor1.GetActorLocation().X > Actor2.GetActorLocation().Y;
				// if (Actor1.GetActorLocation().Z != Actor2.GetActorLocation().Z) return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
				return Actor1.GetActorLocation().Z > Actor2.GetActorLocation().Z;
			});
	}
}